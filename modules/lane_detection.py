"""
Module C: Lane Detection
Algorithm: Ultra-Fast Lane Detection (UFLD) - TuSimple ResNet18 Backbone
Paper: "Ultra Fast Structure-aware Deep Lane Detection" (ECCV 2020)
Weights: Pretrained TuSimple Benchmark (tusimple_18.pth)

Verified against the reference implementation (cfzd/Ultra-Fast-Lane-Detection):
  * cls_dim = (griding_num + 1, num_row_anchors, num_lanes)  -> (101, 56, 4)
  * network input 288x800, ImageNet normalisation
  * row anchors are expressed in the 288-pixel input space
  * x is decoded as the expectation over the griding axis, with the extra
    (griding_num) logit acting as "no lane at this row anchor"

Changes vs. the original Version-1 module are marked with "V1-FIX".
"""

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class ParsingNet(nn.Module):
    """UFLD parsing network with a ResNet-18 backbone (unchanged architecture)."""

    def __init__(
        self,
        size: Tuple[int, int] = (config.UFLD_INPUT_HEIGHT, config.UFLD_INPUT_WIDTH),
        cls_dim: Tuple[int, int, int] = (
            config.UFLD_NUM_GRIDDING + 1,
            config.UFLD_NUM_CLS_PER_LANE,
            config.UFLD_NUM_LANES,
        ),
    ):
        super().__init__()
        self.size = size
        self.cls_dim = cls_dim
        self.total_dim = int(np.prod(cls_dim))

        resnet = models.resnet18(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim),
        )
        self.pool = nn.Conv2d(512, 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fea = self.layer4(x)
        fea = self.pool(fea).view(-1, 1800)
        return self.cls(fea).view(-1, *self.cls_dim)


class LaneTemporalFilter:
    """
    V1-FIX: per-lane temporal smoothing.

    Lanes are associated frame-to-frame by their bottom-row x position; each
    boundary keeps an exponential moving average of its decoded polyline.
    This removes the frame-to-frame jitter of independent per-row argmax
    decoding without lagging real curvature changes more than a few frames.
    """

    def __init__(self, alpha: float = 0.6, max_age: int = 8, assoc_px: int = 120):
        self.alpha = alpha
        self.max_age = max_age
        self.assoc_px = assoc_px
        self.tracks: List[Dict] = []  # {"pts": [(x, y)...], "age": int}

    def reset(self):
        self.tracks = []

    def update(self, lanes: Sequence[Sequence[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        outputs: List[List[Tuple[int, int]]] = []
        live = [l for l in lanes if len(l) >= 2]
        used = set()

        for pts in live:
            bottom = pts[-1][0]
            best, best_d = None, None
            for idx, tr in enumerate(self.tracks):
                if idx in used:
                    continue
                d = abs(tr["bottom"] - bottom)
                if d <= self.assoc_px and (best_d is None or d < best_d):
                    best, best_d = idx, d
            if best is None:
                smoothed = list(pts)
                self.tracks.append({"pts": smoothed, "bottom": bottom, "age": 0})
                used.add(len(self.tracks) - 1)
            else:
                prev = self.tracks[best]["pts"]
                smoothed = self._blend(prev, pts)
                self.tracks[best] = {"pts": smoothed, "bottom": smoothed[-1][0], "age": 0}
                used.add(best)
            outputs.append(smoothed)

        self.tracks = [t for i, t in enumerate(self.tracks) if i in used or t["age"] < self.max_age]
        for i, t in enumerate(self.tracks):
            if i not in used:
                t["age"] += 1

        return outputs

    def _blend(self, prev, cur):
        # Match by y: both lists are bottom-anchored row samples with ~equal spacing.
        prev_by_y = {y: x for x, y in prev}
        out = []
        for x, y in cur:
            if y in prev_by_y:
                x = self.alpha * x + (1.0 - self.alpha) * prev_by_y[y]
            out.append((int(round(x)), int(y)))
        return out


class UltraFastLaneDetector:
    """
    UFLD inference pipeline: preprocess -> network -> decode -> ego corridor.

    V1-FIX summary:
      * strict checkpoint loading (a silent shape mismatch can no longer pass)
      * selectable preprocessing profile instead of hard-coded stretching
      * reference-faithful griding decode (bin centres 0.5..99.5, argmax-based
        non-existence test instead of an uncalibrated probability threshold)
      * ego boundaries chosen by geometry, not by hard-coded lane indices
      * corridor built on a shared y-grid with width/monotonicity sanity checks
      * optional temporal smoothing
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = config.DEVICE,
        min_lane_conf: float = config.LANE_EXISTENCE_THRESHOLD,
        min_lane_points: int = 5,
        preprocess: str = config.LANE_PREPROCESS,
        temporal: bool = config.LANE_TEMPORAL_SMOOTHING,
        model_key: str = config.LANE_MODEL,
    ):
        if model_key not in config.LANE_PROFILES:
            raise ValueError(f"unknown lane model profile '{model_key}'; "
                             f"choose from {sorted(config.LANE_PROFILES)}")
        self.model_key = model_key
        self.profile = config.LANE_PROFILES[model_key]
        self.model_path = model_path or self.profile["weights"]
        self.device = device
        self.row_anchors = np.asarray(self.profile["anchors"], dtype=np.float32)
        self.num_gridding = int(self.profile["griding"])
        self.lane_colors = config.LANE_COLORS
        self.min_lane_conf = min_lane_conf
        self.min_lane_points = min_lane_points
        self.preprocess = preprocess
        self.temporal = temporal
        self.filter = LaneTemporalFilter() if temporal else None
        self.last_corridor_established = False
        self.last_corridor_diagnostics: Dict[str, object] = {}
        self._last_shape: Tuple[int, int] = (720, 1280)

        self.model: Optional[ParsingNet] = None
        self.is_loaded = False
        self.load_error: Optional[str] = None

        self._load_model()

    # ------------------------------------------------------------------ model
    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.load_error = f"weights not found: {self.model_path}"
            print(f"[Module C - Error] {self.load_error}")
            print("Lane detection disabled; the pipeline continues without lanes.")
            return

        try:
            net = ParsingNet(cls_dim=(self.num_gridding + 1,
                                      int(self.profile["rows"]),
                                      int(self.profile["lanes"])))
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            state_dict = {(k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()}

            # V1-FIX: strict load. The previous strict=False would silently skip
            # any tensor whose name or shape changed, producing plausible-looking
            # but meaningless lanes.
            net.load_state_dict(state_dict, strict=True)
            net.to(self.device).eval()
            self.model = net
            self.is_loaded = True
            print(f"[Module C] UFLD '{self.model_key}' weights loaded and verified (strict) from {self.model_path}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            self.load_error = f"{type(exc).__name__}: {exc}"
            print(f"[Module C - Error] Failed to load UFLD model: {self.load_error}")
            self.is_loaded = False

    def reset(self):
        if self.filter is not None:
            self.filter.reset()

    # ------------------------------------------------------------ preprocess
    def _preprocess(self, frame_bgr: np.ndarray):
        """
        Returns (tensor, mapper) where mapper converts network-space (x, y)
        coordinates back to original-image pixels.

        Profiles:
          stretch  - force 800x288 (what the reference demo does; default)
          letterbox- preserve aspect ratio and pad right/bottom
          crop     - drop the top LANE_CROP_TOP fraction, then stretch
        """
        h_orig, w_orig = frame_bgr.shape[:2]
        in_w, in_h = config.UFLD_INPUT_WIDTH, config.UFLD_INPUT_HEIGHT

        if self.preprocess == "crop" and config.LANE_CROP_TOP > 0:
            top = int(round(h_orig * config.LANE_CROP_TOP))
            src = frame_bgr[top:]
            src_h, src_w = src.shape[:2]
            img = cv2.resize(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), (in_w, in_h))
            sx, sy = src_w / float(in_w), src_h / float(in_h)
            mapper = lambda x, y: (x * sx, top + y * sy)  # noqa: E731
        elif self.preprocess == "letterbox":
            scale = min(in_w / float(w_orig), in_h / float(h_orig))
            new_w, new_h = max(1, int(round(w_orig * scale))), max(1, int(round(h_orig * scale)))
            resized = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (new_w, new_h))
            canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
            canvas[:new_h, :new_w] = resized
            img = canvas
            mapper = lambda x, y: (x / scale, y / scale)  # noqa: E731
        else:  # stretch
            img = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (in_w, in_h))
            sx, sy = w_orig / float(in_w), h_orig / float(in_h)
            mapper = lambda x, y: (x * sx, y * sy)  # noqa: E731

        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        return tensor, mapper

    # ---------------------------------------------------------------- decode
    def _decode(self, out: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        Decode the (101, 56, 4) logit tensor into per-lane point lists in
        network coordinates (x in 0..800, y in 0..288).

        x = expectation over the griding axis of softmax(logits[:-1]);
        an anchor is rejected when argmax over all 101 logits selects the
        non-existence bin, matching the reference implementation.
        """
        logits = torch.from_numpy(out).float()
        prob = torch.softmax(logits, dim=0).numpy()
        grid_prob = torch.softmax(logits[:-1], dim=0).numpy()
        nonexist_prob = prob[self.num_gridding]  # (56, 4)
        argmax = out.argmax(axis=0)              # (56, 4)
        centres = (np.arange(self.num_gridding) + 0.5) * (config.UFLD_INPUT_WIDTH / self.num_gridding)
        loc = (grid_prob * centres.reshape(-1, 1, 1)).sum(axis=0)  # (56, 4)

        lanes: List[List[Tuple[int, int]]] = []
        for lane_idx in range(int(self.profile["lanes"])):
            pts: List[Tuple[int, int]] = []
            for r_idx in range(len(self.row_anchors)):
                if argmax[r_idx, lane_idx] >= self.num_gridding:
                    continue  # model says: no lane here
                if (1.0 - nonexist_prob[r_idx, lane_idx]) < self.min_lane_conf:
                    continue
                x = float(loc[r_idx, lane_idx])
                y = float(self.row_anchors[r_idx])
                if 0 <= x <= config.UFLD_INPUT_WIDTH:
                    pts.append((int(round(x)), int(round(y))))
            lanes.append(pts)
        return lanes

    # ---------------------------------------------------------------- detect
    def detect(
        self,
        frame_bgr: np.ndarray,
        obstacle_boxes: Optional[List[Sequence[float]]] = None,
    ) -> Tuple[List[List[Tuple[int, int]]], float]:
        """
        Detect lanes. Returns (lanes, inference_time_ms) where each lane is a
        list of (x, y) points in ORIGINAL image coordinates, top -> bottom.
        """
        if not self.is_loaded or self.model is None:
            return [[] for _ in range(int(self.profile["lanes"]))], 0.0

        t0 = time.perf_counter()
        h_orig, w_orig = frame_bgr.shape[:2]
        tensor, mapper = self._preprocess(frame_bgr)

        with torch.no_grad():
            out = self.model(tensor)[0].cpu().numpy()

        raw_lanes = self._decode(out)

        lanes: List[List[Tuple[int, int]]] = []
        for lane_idx, raw_pts in enumerate(raw_lanes):
            mapped: List[Tuple[int, int]] = []
            for x_net, y_net in raw_pts:
                x, y = mapper(float(x_net), float(y_net))
                x, y = int(round(x)), int(round(y))
                if not (0 <= x < w_orig and 0 <= y < h_orig):
                    continue
                if self._is_occluded(x, y, obstacle_boxes):
                    continue
                mapped.append((x, y))

            # Drop lateral teleports but keep legitimate curvature: a valid lane
            # moves outward smoothly, so the allowed step scales with |dy|.
            filtered: List[Tuple[int, int]] = []
            for pt in mapped:
                if not filtered:
                    filtered.append(pt)
                    continue
                px, py = filtered[-1]
                allowed = max(0.03 * w_orig, 1.2 * abs(pt[1] - py) + 0.02 * w_orig)
                if abs(pt[0] - px) <= allowed:
                    filtered.append(pt)

            lanes.append(filtered if len(filtered) >= self.min_lane_points else [])

        if self.filter is not None:
            lanes = self._restore_slots(lanes, self.filter.update(lanes))

        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        return lanes, inference_time_ms

    @staticmethod
    def _is_occluded(x: int, y: int, boxes: Optional[Sequence[Sequence[float]]]) -> bool:
        if not boxes:
            return False
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            # Only suppress points well inside a vehicle body; paint at the box
            # border is usually still visible and remains a valid lane sample.
            if x1 + 2 <= x <= x2 - 2 and y1 + 0.4 * (y2 - y1) <= y <= y2 - 2:
                return True
        return False

    def _restore_slots(self, original, smoothed):
        """Temporal filter drops/reorders lanes; restore the per-profile slot layout."""
        slots: List[List[Tuple[int, int]]] = [[] for _ in range(int(self.profile["lanes"]))]
        pool = [l for l in smoothed if len(l) >= 2]
        for idx, lane in enumerate(original):
            if len(lane) < 2:
                continue
            bottom_x = lane[-1][0]
            best, best_d = None, None
            for cand in pool:
                d = abs(cand[-1][0] - bottom_x)
                if best is None or d < best_d:
                    best, best_d = cand, d
            if best is not None and best_d is not None and best_d <= 120:
                slots[idx] = list(best)
                pool.remove(best)
        return slots

    # -------------------------------------------------------------- corridor
    def _select_ego_boundaries(self, lanes):
        """
        V1-FIX: pick the pair of lanes that brackets the ego vehicle (image
        centre at the bottom row) instead of assuming slots 1 and 2.
        Returns (left, right, diagnostics).
        """
        h, w = self._last_shape
        bottom_tol = 0.6 * h
        cands = []
        for idx, pts in enumerate(lanes):
            if len(pts) < self.min_lane_points:
                continue
            low = [(x, y) for x, y in pts if y >= bottom_tol]
            if not low:
                continue
            bx = float(np.median([x for x, _ in low]))
            cands.append({"idx": idx, "bottom_x": bx, "pts": pts})

        diag = {"candidates": [(c["idx"], round(c["bottom_x"])) for c in cands]}
        if len(cands) < 2:
            diag["reason"] = "fewer than two usable boundaries"
            return None, None, diag

        centre = w / 2.0
        left = [c for c in cands if c["bottom_x"] <= centre]
        right = [c for c in cands if c["bottom_x"] > centre]
        left.sort(key=lambda c: -c["bottom_x"])
        right.sort(key=lambda c: c["bottom_x"])

        if left and right:
            l, r = left[0], right[0]
        else:
            # No boundary straddles the centre: fall back to the closest pair.
            cands.sort(key=lambda c: c["bottom_x"])
            gaps = [(cands[i + 1]["bottom_x"] - cands[i]["bottom_x"], i) for i in range(len(cands) - 1)]
            gaps = [g for g in gaps if g[0] > 0]
            if not gaps:
                diag["reason"] = "no distinct boundary pair"
                return None, None, diag
            gaps.sort(key=lambda g: -g[0])
            l, r = cands[gaps[0][1]], cands[gaps[0][1] + 1]
            diag["fallback"] = "widest pair (centre not bracketed)"

        width = r["bottom_x"] - l["bottom_x"]
        min_w = config.LANE_MIN_WIDTH_FRAC * w
        max_w = config.LANE_MAX_WIDTH_FRAC * w
        diag.update({"left_idx": l["idx"], "right_idx": r["idx"], "bottom_width_px": round(width, 1)})
        if not (min_w <= width <= max_w):
            diag["reason"] = f"implausible lane width {width:.0f}px (allowed {min_w:.0f}-{max_w:.0f})"
            return None, None, diag

        return l["pts"], r["pts"], diag

    @staticmethod
    def _common_grid(left, right, samples: int = 24):
        """
        V1-FIX: resample both boundaries on one shared y-grid before building the
        polygon. Concatenating two independently sampled series produced
        self-intersecting fills and a spurious horizontal bottom edge.
        """
        ly = np.array([p[1] for p in left], dtype=np.float32)
        ry = np.array([p[1] for p in right], dtype=np.float32)
        y_lo = max(ly.min(), ry.min())
        y_hi = min(ly.max(), ry.max())
        if y_hi - y_lo < 40:
            return None
        ys = np.linspace(y_lo, y_hi, samples)
        lx = np.interp(ys, ly, np.array([p[0] for p in left], dtype=np.float32))
        rx = np.interp(ys, ry, np.array([p[0] for p in right], dtype=np.float32))
        if np.any(rx <= lx):  # boundaries crossed -> refuse to fill garbage
            return None
        return np.stack([np.stack([lx, ys], axis=1), np.stack([rx, ys], axis=1)])

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        lanes: List[List[Tuple[int, int]]],
        draw_drive_area: bool = True,
    ) -> np.ndarray:
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        self._last_shape = (h, w)
        self.last_corridor_established = False
        self.last_corridor_diagnostics = {}

        if draw_drive_area:
            left, right, diag = self._select_ego_boundaries(lanes)
            self.last_corridor_diagnostics = diag
            if left is not None and right is not None:
                grid = self._common_grid(left, right)
                if grid is None:
                    diag["reason"] = "boundaries do not overlap or cross"
                else:
                    poly_pts = np.concatenate([grid[0], grid[1][::-1]]).astype(np.int32)
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [poly_pts], (0, 180, 0))
                    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
                    self.last_corridor_established = True
                    diag["corridor"] = "established"

        for l_idx, pts in enumerate(lanes):
            if len(pts) < 2:
                continue
            color = self.lane_colors[l_idx % len(self.lane_colors)]
            cv2.polylines(
                annotated,
                [np.array(pts, dtype=np.int32)],
                isClosed=False,
                color=color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            for pt in pts:
                cv2.circle(annotated, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return annotated
