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

# Fewest rows two boundaries must share before a corridor can be sampled at all.
# Below this the two polylines barely meet and interpolating between them would
# invent the span, so the pair is refused for lack of overlap.
COMMON_GRID_MIN_ROWS = 40


def paint_support_mask(
    frame_bgr: np.ndarray,
    local_win: Optional[int] = None,
    delta: Optional[int] = None,
) -> np.ndarray:
    """
    Boolean mask of pixels that look like road paint.

    An absolute brightness cut is not usable here: 110/255 is asphalt in a
    sunlit frame and paint under headlights at night. Each pixel is therefore
    compared with the median brightness of its own neighbourhood, so a bright
    marking registers on both sunlit and night frames.

    Calibrated on the 135 hand-verified GT points in labels/lane/clips_gt.json
    (wet daylight mountain road, dusk motorway, night motorway): the mask finds
    paint under 98% of them, per boundary 14/15 fully supported. That means a
    negative answer means "no paint here" rather than "threshold too strict",
    which is what lets it trim phantom lane segments. See
    scripts/calibrate_paint_support.py.
    """
    local_win = local_win or config.LANE_PAINT_LOCAL_WIN
    delta = config.LANE_PAINT_DELTA if delta is None else delta
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    val_f = val.astype(np.float32)
    local = cv2.blur(val_f, (int(local_win), int(local_win)))
    bright = val_f >= np.maximum(70.0, local + float(delta))
    white = bright & (sat <= 110)
    yellow = (hue >= 15) & (hue <= 45) & (sat >= 90) & (val_f >= 80)
    return white | yellow


def boundary_support(
    mask: np.ndarray,
    pts: Sequence[Sequence[float]],
    half_win: Optional[int] = None,
) -> List[bool]:
    """
    Per-point "is there paint under this boundary point?" using a small window
    (5 rows tall, +-half_win wide). The window tolerates the few-pixel offset
    between a decoded centreline and the paint it sits on.
    """
    half = config.LANE_PAINT_HALF_WIN if half_win is None else int(half_win)
    h, w = mask.shape[:2]
    out: List[bool] = []
    for pt in pts:
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        y0, y1 = max(0, y - 2), min(h, y + 3)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        out.append(bool(mask[y0:y1, x0:x1].any()) if y1 > y0 and x1 > x0 else False)
    return out


def close_short_gaps(flags: Sequence[bool], max_gap: int) -> List[bool]:
    """
    Bridge runs of False no longer than max_gap that are flanked by True.

    A single missed sample in the middle of a painted line is a sampling
    artefact, not evidence that the paint stops (measured: on frame 20 of clip F
    the yellow line is detected continuously but one of 27 samples lands 60px
    off and registers nothing). Longer runs are left alone - those are where the
    boundary genuinely leaves the paint.
    """
    out = list(bool(f) for f in flags)
    if max_gap <= 0:
        return out
    i = 0
    while i < len(out):
        if out[i]:
            i += 1
            continue
        j = i
        while j < len(out) and not out[j]:
            j += 1
        if i > 0 and j < len(out) and (j - i) <= max_gap:
            for k in range(i, j):
                out[k] = True
        i = j
    return out


def longest_true_run(flags: Sequence[bool]) -> Tuple[int, int]:
    """(start, end) inclusive of the longest run of True, or (-1, -1)."""
    best = (-1, -1)
    start = None
    for i, ok in enumerate(flags):
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == len(flags) - 1):
            end = i if ok else i - 1
            if end - start > best[1] - best[0]:
                best = (start, end)
            start = None
    return best


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
        device: Optional[str] = None,
        min_lane_conf: Optional[float] = None,
        min_lane_points: Optional[int] = None,
        preprocess: Optional[str] = None,
        temporal: Optional[bool] = None,
        model_key: Optional[str] = None,
    ):
        # V1-FIX: every setting below is resolved at CALL time. Binding them as
        # default arguments evaluated config at import, so a caller that set
        # config.LANE_MODEL after importing this module (main.py did, from
        # --lane-model, and --lane-preprocess with it) had its choice silently
        # ignored and always got the tusimple/stretch profile.
        device = device or config.DEVICE
        min_lane_conf = (config.LANE_EXISTENCE_THRESHOLD if min_lane_conf is None
                         else min_lane_conf)
        min_lane_points = 5 if min_lane_points is None else min_lane_points
        preprocess = preprocess or config.LANE_PREPROCESS
        temporal = config.LANE_TEMPORAL_SMOOTHING if temporal is None else temporal
        model_key = model_key or config.LANE_MODEL
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
        # V1-FIX: record the frame shape here, not only in annotate_frame().
        # _select_ego_boundaries() reads _last_shape to split left/right at the
        # image centre; a caller that runs detect() first (scripts/eval_lane.py)
        # used to get the 1280x720 default on a 1920x1080 frame, i.e. a centre of
        # x=640 instead of x=960, and a bottom band starting at y=432 instead of
        # y=648 -- which changes which boundaries qualify as the ego pair.
        self._last_shape = (h_orig, w_orig)
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
                if not self._in_roi(y, h_orig):
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
    def _in_roi(y: int, height: int) -> bool:
        """Optional hard region of interest (config.LANE_ROI_*_FRAC).

        Default is the whole frame. This exists so an operator can exclude a hood
        band outright; the paint test below handles the hood automatically, by
        checking whether the pixels under a bottom-row point are actually paint
        rather than assuming everything above the bumper is road.
        """
        top = config.LANE_ROI_TOP_FRAC * height
        bottom = config.LANE_ROI_BOTTOM_FRAC * height
        return top <= y < bottom

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

        bottom_x is the x of the boundary at ITS OWN lowest point, not the median
        of everything in the lower 40% of the frame. The old form let a boundary
        that exists only high in the image (a decoded line in the field, above
        the road) be scored by a pixel position that was never its own bottom -
        measured on frame 20 of clip F, the old code took a phantom lane ending
        at (1764, 1020) and used it as the ego lane's right edge.
        """
        h, w = self._last_shape
        cands = []
        for idx, pts in enumerate(lanes):
            if len(pts) < self.min_lane_points:
                continue
            pts_sorted = sorted(pts, key=lambda p: p[1])
            y_bot = pts_sorted[-1][1]
            # median over the last few points only, so one stray sample cannot
            # define the bottom position of the whole boundary
            band = [x for x, y in pts_sorted if y >= y_bot - 0.02 * h]
            if not band:
                continue
            bx = float(np.median(band))
            cands.append({"idx": idx, "bottom_x": bx, "bottom_y": y_bot, "pts": pts})

        diag = {"candidates": [(c["idx"], round(c["bottom_x"]), int(c["bottom_y"]))
                              for c in cands]}
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

        # Width at the lowest row BOTH boundaries describe. Subtracting two
        # bottom_x values that were measured at different rows inflates the width
        # whenever one boundary stops early - measured on D02 and F01 that turned
        # a real ego lane into a 1447/1928px "implausible" one.
        width = r["bottom_x"] - l["bottom_x"]
        shared = self._common_grid(l["pts"], r["pts"])
        if shared is not None:
            width = float(shared[2][-1] - shared[1][-1])
            diag["width_row"] = int(round(float(shared[0][-1])))
        min_w = config.LANE_MIN_WIDTH_FRAC * w
        max_w = config.LANE_MAX_WIDTH_FRAC * w
        diag.update({"left_idx": l["idx"], "right_idx": r["idx"], "bottom_width_px": round(width, 1),
                     "left_bottom_y": int(l["bottom_y"]), "right_bottom_y": int(r["bottom_y"])})
        if not (min_w <= width <= max_w):
            diag["reason"] = f"implausible lane width {width:.0f}px (allowed {min_w:.0f}-{max_w:.0f})"
            return None, None, diag

        return l["pts"], r["pts"], diag

    @staticmethod
    def _shared_span(left, right) -> Tuple[float, float]:
        """Rows covered by BOTH boundary polylines, ignoring paint entirely."""
        ly = [p[1] for p in left]
        ry = [p[1] for p in right]
        return max(min(ly), min(ry)), min(max(ly), max(ry))

    @staticmethod
    def _common_grid(left, right, samples: int = 60):
        """
        V1-FIX: resample both boundaries on one shared y-grid before building the
        polygon. Concatenating two independently sampled series produced
        self-intersecting fills and a spurious horizontal bottom edge.

        Returns (ys, lx, rx) or None. Only the rows both boundaries actually
        cover are interpolated, so nothing here is extrapolated.
        """
        ly = np.array([p[1] for p in left], dtype=np.float32)
        ry = np.array([p[1] for p in right], dtype=np.float32)
        y_lo = max(ly.min(), ry.min())
        y_hi = min(ly.max(), ry.max())
        if y_hi - y_lo < COMMON_GRID_MIN_ROWS:
            return None
        ys = np.linspace(y_lo, y_hi, samples)
        lx = np.interp(ys, ly, np.array([p[0] for p in left], dtype=np.float32))
        rx = np.interp(ys, ry, np.array([p[0] for p in right], dtype=np.float32))
        if np.any(rx <= lx):  # boundaries crossed -> refuse to fill garbage
            return None
        return ys, lx, rx

    def _corridor(self, frame_bgr: np.ndarray, left, right):
        """
        Build the ego-corridor polygon, or return None plus the reason.

        A corridor is a claim about the lane the vehicle is IN, so it has to
        satisfy three things that the old version never checked:

        1. Every published row has road paint under BOTH boundaries. Rows
           without paint are trimmed, and only the longest contiguous run of
           supported rows survives - a lane cannot be "supported" below a gap.
           This is what removes the field/hillside polygon: on frame 20 of clip
           F the selected right edge passed through grass.
        2. The surviving corridor is at least LANE_CORRIDOR_MIN_ROWS_FRAC of the
           frame tall, AND the two decoded boundaries overlap over at least that
           much before trimming. Trimming can only shorten the span, so a pair
           that never shares enough rows is refused as insufficient overlap
           instead of being blamed on the paint test.
        3. The corridor reaches the part of the image where this vehicle's own
           lane is - its lowest row is no more than
           LANE_CORRIDOR_MAX_BOTTOM_GAP_FRAC above the frame bottom - and the
           camera axis (centre column) lies between its two edges there. A
           corridor floating hundreds of pixels above the bonnet is not this
           vehicle's lane, even if its edges are locally straight.

           Note this is measured after trimming, so a lane whose bottom rows sit
           on the bonnet is trimmed up to the paint and still published; only a
           corridor that stops high in the frame is refused.

        Returns (polygon_pts_int32 or None, diagnostics).
        """
        h, w = frame_bgr.shape[:2]
        # How many rows the two DECODED boundaries cover together, measured
        # before any paint trimming. This is the ceiling on corridor height, so
        # it is what distinguishes "the model never saw both edges over a long
        # enough stretch" from "the paint test cut the span short".
        span = self._shared_span(left, right)
        overlap = max(0.0, span[1] - span[0])
        need = config.LANE_CORRIDOR_MIN_ROWS_FRAC * h
        grid = self._common_grid(left, right)
        if grid is None:
            # Either the edges are disjoint/crossed, or they share too few rows.
            # Only the second case is an overlap shortfall worth naming; the
            # first keeps the wording it always had.
            if overlap <= 0 or overlap >= COMMON_GRID_MIN_ROWS:
                return None, {"reason": "boundaries do not overlap or cross",
                              "overlap_rows": int(round(overlap))}
            return None, {
                "reason": (f"boundaries share only {overlap:.0f} rows "
                           f"(need {COMMON_GRID_MIN_ROWS:.0f} to sample one)"),
                "overlap_rows": int(round(overlap)),
            }
        ys, lx, rx = grid
        diag = {
            "rows": int(len(ys)),
            "y_lo": int(round(float(ys.min()))),
            "y_hi": int(round(float(ys.max()))),
            "overlap_rows": int(round(overlap)),
        }

        if config.LANE_PAINT_SUPPORT:
            mask = paint_support_mask(frame_bgr)
            sup_l = boundary_support(mask, list(zip(lx.tolist(), ys.tolist())))
            sup_r = boundary_support(mask, list(zip(rx.tolist(), ys.tolist())))
            both = [a and b for a, b in zip(sup_l, sup_r)]
            diag["support_left"] = round(float(np.mean(sup_l)), 3)
            diag["support_right"] = round(float(np.mean(sup_r)), 3)
            diag["support_both"] = round(float(np.mean(both)), 3)
            both = close_short_gaps(both, config.LANE_PAINT_MAX_GAP_POINTS)
            start, end = longest_true_run(both)
            if start < 0:
                diag["reason"] = "no row has paint under both boundaries"
                return None, diag
            if end - start + 1 < len(both):
                diag["trimmed_rows"] = int(len(both) - (end - start + 1))
            ys, lx, rx = ys[start:end + 1], lx[start:end + 1], rx[start:end + 1]
            diag["y_lo"] = int(round(float(ys.min())))
            diag["y_hi"] = int(round(float(ys.max())))
            diag["rows_kept"] = int(len(ys))

        height = float(ys.max() - ys.min())
        diag["height_frac"] = round(height / h, 3)
        if height < need:
            if overlap < need:
                # Even with perfect paint neither edge reaches high enough for
                # the two to share the required rows: this is a coverage limit
                # of the checkpoint, not a paint-filter effect.
                diag["reason"] = (f"boundaries share only {overlap:.0f} rows "
                                  f"(need {need:.0f} for a corridor)")
            else:
                diag["reason"] = (f"supported corridor too short ({height:.0f}px, "
                                  f"need {need:.0f}px)")
            return None, diag

        poly = np.concatenate([
            np.stack([lx, ys], axis=1),
            np.stack([rx, ys], axis=1)[::-1],
        ]).astype(np.int32)

        if config.LANE_REQUIRE_EGO_ANCHOR:
            bottom_gap = (h - float(ys.max())) / h
            diag["bottom_gap_frac"] = round(bottom_gap, 3)
            if bottom_gap > config.LANE_CORRIDOR_MAX_BOTTOM_GAP_FRAC:
                diag["reason"] = (f"corridor stops {bottom_gap:.0%} of the frame above the "
                                  f"bottom, i.e. not the lane this vehicle is in")
                return None, diag
            centre = w * 0.5
            width_b = float(rx[-1]) - float(lx[-1])
            tol = config.LANE_EGO_AXIS_TOL_FRAC * max(1.0, width_b)
            if not (float(lx[-1]) - tol < centre < float(rx[-1]) + tol):
                diag["reason"] = ("camera axis is not inside this corridor at its bottom "
                                  f"row (axis {centre:.0f}px, edges "
                                  f"{lx[-1]:.0f}-{rx[-1]:.0f}px)")
                return None, diag
            diag["axis_offset_px"] = round(float(centre - 0.5 * (lx[-1] + rx[-1])), 1)

        return poly, diag

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        lanes: List[List[Tuple[int, int]]],
        draw_drive_area: bool = True,
    ) -> np.ndarray:
        """
        Draw lane boundaries and, when the evidence supports it, the ego corridor.

        Only lane segments that sit on paint are drawn: a decoded line running up
        a hillside is still a decoded line, and drawing it solid implied a
        detection the pixels do not support. Unsupported points are reported in
        last_corridor_diagnostics as "unsupported_frac" instead of being hidden.
        """
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        self._last_shape = (h, w)
        self.last_corridor_established = False
        self.last_corridor_diagnostics = {}

        mask = paint_support_mask(annotated) if config.LANE_PAINT_SUPPORT else None

        if draw_drive_area:
            left, right, diag = self._select_ego_boundaries(lanes)
            self.last_corridor_diagnostics = diag
            if left is not None and right is not None:
                poly_pts, cdiag = self._corridor(annotated, left, right)
                diag.update(cdiag)
                if poly_pts is not None:
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [poly_pts], (0, 180, 0))
                    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
                    self.last_corridor_established = True
                    diag["corridor"] = "established"

        for l_idx, pts in enumerate(lanes):
            if len(pts) < 2:
                continue
            color = self.lane_colors[l_idx % len(self.lane_colors)]
            if mask is None:
                segments, unsupported, why = [pts], 0, "paint test disabled"
            else:
                support = boundary_support(mask, pts)
                unsupported = int(sum(1 for ok in support if not ok))
                segments, why = self._evidence_segment(pts, support, h)
                self.last_corridor_diagnostics.setdefault("lane_support", {})[l_idx] = {
                    "points": len(pts),
                    "supported": len(pts) - unsupported,
                }
            self.last_corridor_diagnostics.setdefault("lane_draw", {})[l_idx] = why
            for seg in segments:
                if len(seg) < 2:
                    continue
                cv2.polylines(
                    annotated,
                    [np.array(seg, dtype=np.int32)],
                    isClosed=False,
                    color=color,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                for pt in seg:
                    cv2.circle(annotated, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            if unsupported:
                self.last_corridor_diagnostics.setdefault("unsupported_frac", {})[l_idx] = round(
                    unsupported / float(len(pts)), 3)

        return annotated

    @staticmethod
    def _evidence_segment(
        pts: Sequence[Tuple[int, int]],
        support: Sequence[bool],
        height: int,
        min_points: int = 2,
    ) -> Tuple[List[List[Tuple[int, int]]], str]:
        """
        The stretch of a boundary that is both on paint and connected to the
        ego rows.

        Two things were wrong with drawing every decoded point:

        * the top of a lane is where the decoder extrapolates least reliably, and
          on clip F it ran up a hillside;
        * the paint test cannot tell white paint from a white fence post or a
          utility pole, so a short bright string in a field can pass locally.

        Connectivity decides the second case: a lane on the road surface always
        reaches the bottom of a forward-facing frame, so only the supported run
        that contains the lowest supported point inside the ego band is drawn.
        Everything above a gap, and everything that never comes near the vehicle,
        is left out - and the reason is recorded, not silently dropped.
        """
        band_top = height * (1.0 - config.LANE_CORRIDOR_MAX_BOTTOM_GAP_FRAC)
        closed = close_short_gaps(support, config.LANE_PAINT_MAX_GAP_POINTS)
        runs: List[Tuple[int, int]] = []
        start = None
        for i, ok in enumerate(closed):
            if ok and start is None:
                start = i
            if start is not None and (not ok or i == len(closed) - 1):
                runs.append((start, i if ok else i - 1))
                start = None
        # A lane this vehicle is in has paint somewhere in the ego band. The run
        # drawn is the longest such run: a single supported sample at the very
        # bottom is usually the decoder row landing a few pixels off the paint,
        # and it must not shadow the 25-point stretch directly above it.
        candidates = [r for r in runs if pts[r[1]][1] >= band_top]
        if not candidates:
            return [], "no paint evidence near the ego rows"
        s, e = max(candidates, key=lambda r: r[1] - r[0])
        run = list(pts[s:e + 1])
        if len(run) < min_points:
            return [], "supported run too short to draw"
        return [run], "drawn from the ego rows upward"
