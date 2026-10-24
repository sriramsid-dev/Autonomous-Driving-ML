"""
Module B (Part 2): Traffic Light State Classification

V1-FIX: replaces whole-crop HSV scoring with a lamp-blob + circular-hue
classifier, plus temporal voting.

Why the previous approach failed (measured, see EVALUATION.md):
  * the spatial prior assumed the crop is a complete 3-lamp housing, but the
    COCO detector returns the housing *plus* background (measured boxes up to
    369x461 px), so sky / foliage / brick hue was scored as if it were a lamp;
  * a crop tight around one lit lamp was penalised by that same prior
    (w_red falls to 0.1 at the bottom row of the crop);
  * at night the lit core blows out to white, so saturation -> 0 and hue is
    undefined; fixed hue ranges then lose the lamp entirely.

The classifier below is still deterministic (no trained weights), but it works
on the *lit lamp blob* instead of the whole box, measures hue circularly on the
saturated halo of that blob, and only emits a state after temporal agreement.
It is a baseline, not a learned classifier: `capability` is reported so callers
cannot mistake it for a trained model.
"""

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# Reference hues in OpenCV's 8-bit H scale (0-179)
_HUE_CENTRES = {"red": 0.0, "yellow": 25.0, "green": 60.0}
# Per-state tolerance: LED "green" signals are often cyan (H ~ 80-90), while red
# and yellow are tightly separated and need a narrow window.
_HUE_TOLERANCE_BY_STATE = {"red": 20.0, "yellow": 18.0, "green": 35.0}
_HUE_TOLERANCE = max(_HUE_TOLERANCE_BY_STATE.values())


def _circular_delta(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


class TrafficLightClassifier:
    """Classifies a traffic-light crop into red / yellow / green / unknown."""

    capability = "heuristic-baseline (lamp blob + circular hue + temporal voting)"

    def __init__(
        self,
        min_crop_area: int = config.TL_MIN_CROP_AREA,
        min_width: int = config.TL_MIN_WIDTH,
        min_height: int = config.TL_MIN_HEIGHT,
        ambiguity_threshold: float = config.TL_MIN_STATE_CONF,
        temporal_window: int = config.TL_TEMPORAL_WINDOW,
        temporal_min_votes: int = config.TL_TEMPORAL_MIN_VOTES,
    ):
        self.min_crop_area = min_crop_area
        self.min_width = min_width
        self.min_height = min_height
        self.ambiguity_threshold = ambiguity_threshold
        self.temporal_window = temporal_window
        self.temporal_min_votes = temporal_min_votes

        self.state_colors = {
            "red": (0, 0, 255),
            "yellow": (0, 215, 255),
            "green": (0, 255, 0),
            "unknown": (160, 160, 160),
        }
        self._votes: Dict[object, Deque[str]] = {}
        self.last_debug: Dict[str, object] = {}

    # ------------------------------------------------------------------ blob
    @staticmethod
    def _upsample(crop_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Upscale small crops so blob geometry is measurable; return scale used."""
        h, w = crop_bgr.shape[:2]
        shortest = min(h, w)
        if shortest >= 48:
            return crop_bgr, 1.0
        scale = 48.0 / max(1, shortest)
        return cv2.resize(crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC), scale

    @staticmethod
    def _lit_lamp_mask(bgr: np.ndarray) -> np.ndarray:
        """
        Lit-lamp candidate mask: saturated *or* strongly over-exposed relative to
        the crop's own median brightness (the night-time white core).
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        v_med = float(np.median(v))
        bright = (v >= max(150, 1.45 * v_med)).astype(np.uint8)
        saturated = ((s >= config.TL_BLOB_MIN_SAT) & (v >= config.TL_BLOB_MIN_VALUE)).astype(np.uint8)
        mask = cv2.bitwise_or(bright, saturated)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask

    def _dominant_hue(self, bgr: np.ndarray, mask: np.ndarray):
        """
        Circular hue histogram over the blob's saturated pixels.

        Returns (peak_hue, share, pixel_count) where share is the fraction of
        analysed saturated pixels inside +-tolerance of the peak.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sel = (mask > 0) & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 60)
        n = int(sel.sum())
        if n < config.TL_LAMP_MIN_PIXELS:
            return None, 0.0, n

        hues = hsv[:, :, 0][sel].astype(np.float32)
        hist = np.zeros(36, dtype=np.float32)  # 5-degree bins
        np.add.at(hist, (hues // 5).astype(np.int32) % 36, 1.0)
        # circular smoothing so a hue split across the 0/180 wrap is not lost
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel /= kernel.sum()
        smooth = np.array([np.sum(hist[(np.arange(-2, 3) + i) % 36] * kernel) for i in range(36)])
        peak_bin = int(np.argmax(smooth))
        peak_hue = peak_bin * 5.0 + 2.5
        # Confidence = fraction of analysed lamp pixels whose hue is within
        # tolerance of the peak (a smoothed-histogram peak share is far too
        # pessimistic and rejected most real lamps).
        within = np.array([_circular_delta(float(h), peak_hue) <= _HUE_TOLERANCE for h in hues])
        share = float(within.mean()) if within.size else 0.0
        return peak_hue, share, n

    # -------------------------------------------------------------- classify
    def classify(self, crop_bgr: np.ndarray, key: Optional[object] = None) -> Tuple[str, float]:
        """
        Classify one crop. When `key` is given, the result is subject to temporal
        voting (call once per frame with a stable per-signal key).
        """
        state, conf = self._classify_single(crop_bgr)
        if key is not None:
            state, conf = self._vote(key, state, conf)
        return state, conf

    def _classify_single(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        self.last_debug = {}
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown", 0.0

        h, w = crop_bgr.shape[:2]
        if (h * w) < self.min_crop_area or w < self.min_width or h < self.min_height:
            self.last_debug["reject"] = "crop smaller than configured minimum"
            return "unknown", 0.0

        bgr, scale = self._upsample(crop_bgr)
        hh, ww = bgr.shape[:2]
        mask = self._lit_lamp_mask(bgr)

        # Keep only the largest connected blob plus its saturated halo: this is
        # the lamp itself, not the surrounding scene.
        n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_lab <= 1:
            self.last_debug["reject"] = "no lit region found"
            return "unknown", 0.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        lamp_area = float(areas[best - 1])
        cx, cy = cents[best]

        hue, share, n_px = self._dominant_hue(bgr, mask)
        self.last_debug.update(
            {
                "lamp_pixels": n_px,
                "lamp_area_px": int(lamp_area),
                "lamp_share": round(lamp_area / float(ww * hh), 3),
                "peak_hue": None if hue is None else round(float(hue), 1),
                "hue_share": round(share, 3),
                "upscale": round(scale, 2),
            }
        )
        if hue is None:
            self.last_debug["reject"] = "lit region has too little saturated evidence"
            return "unknown", 0.0
        if n_px < config.TL_LAMP_MIN_PIXELS:
            self.last_debug["reject"] = f"only {n_px} lamp pixels"
            return "unknown", 0.0

        # Normalise distance by the state's tolerance so a cyan-green lamp is
        # still matched to green rather than rejected.
        norm = {name: _circular_delta(hue, c) / _HUE_TOLERANCE_BY_STATE[name] for name, c in _HUE_CENTRES.items()}
        state = min(norm, key=norm.get)
        if norm[state] > 1.0:
            self.last_debug["reject"] = f"hue {hue:.0f} matches no state"
            return "unknown", 0.0

        conf = float(share)

        # Weak spatial veto only when the crop really looks like a multi-lamp
        # housing (tall, with the blob in a clearly contradictory slot).
        blob_h = float(stats[best, cv2.CC_STAT_HEIGHT])
        if hh >= 2.2 * blob_h:
            rel = cy / hh
            if state == "red" and rel > 0.70:
                conf *= 0.55
            elif state == "green" and rel < 0.30:
                conf *= 0.55
            self.last_debug["spatial_rel_y"] = round(float(rel), 2)

        if conf < self.ambiguity_threshold:
            self.last_debug["reject"] = f"low confidence {conf:.2f}"
            return "unknown", conf
        return state, conf

    # ------------------------------------------------------------------ vote
    def _vote(self, key: object, state: str, conf: float) -> Tuple[str, float]:
        hist = self._votes.setdefault(key, deque(maxlen=self.temporal_window))
        hist.append(state if conf >= self.ambiguity_threshold else "unknown")
        if len(hist) < self.temporal_min_votes:
            return "unknown", conf
        counts: Dict[str, int] = {}
        for s in hist:
            counts[s] = counts.get(s, 0) + 1
        best = max(counts, key=counts.get)
        if best == "unknown" or counts[best] < self.temporal_min_votes:
            return "unknown", conf
        return best, conf

    def reset_temporal(self):
        self._votes.clear()

    # -------------------------------------------------------------- annotate
    def annotate(
        self,
        frame_bgr: np.ndarray,
        box: Tuple[float, float, float, float],
        state: str,
        conf: float,
    ) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = self.state_colors.get(state, self.state_colors["unknown"])

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        label = f"TL: {state.upper()} ({conf:.2f})"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_y1 = max(0, y1 - th - baseline - 4)
        badge_y2 = y1
        cv2.rectangle(frame_bgr, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)

        text_color = (255, 255, 255) if state != "yellow" else (0, 0, 0)
        cv2.putText(
            frame_bgr,
            label,
            (x1 + 3, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )
        return frame_bgr
