"""
Lamp classifier v1.4: per-state chromatic components scored by surrounding contrast.

Why the earlier versions failed (all measured on the dev crops, EVALUATION.md 9.1):

  * v1.0 scored the *whole crop* with a spatial prior that assumed a complete
    3-lamp housing. Sky/foliage/brick hue counted as a lamp, and a crop tight
    around one lit lamp was penalised for being tight.
  * v1.1 took the *largest* saturated blob and then computed the hue histogram
    over the *entire* mask instead of that blob. Two measured consequences:
      - the cream/yellow housing is the largest chromatic region in most boxes,
        so its hue won even with a clearly lit green lamp (crop 39: peak hue
        17.5 = amber, from 16 661 pixels of housing);
      - yellow paint and amber glass have the *same* chromaticity
        (`min(R,G)-B` ~0.6 for both), so no colour rule can separate them.
        Unlit housings came back "yellow" at 0.8-1.0 confidence.

v1.4 keeps the deterministic approach but changes the decision. A lit lamp is a
small chromatic region sitting in a *dark bezel* and brighter than it; paint is
a large chromatic region surrounded by more of the same paint. So each candidate
component is scored by

    chroma_strength x min(1, ring_contrast / TL_RING_FULL_CONTRAST)

where ring_contrast is the component's 80th-percentile luminance minus the
median luminance of a ring grown outwards from it. A region that fills the box
in a structural crop is the signal body and is discounted, and a state is only
returned when it beats the best competing state by TL_SCORE_MARGIN, otherwise
the answer is "unknown". The score is heuristic evidence, not a probability.

Regimes: a crop whose short side is <= TL_TIGHT_CROP_SIDE cannot contain a
full 3-lamp housing, so the body discount is skipped there (a detector box
tight on one lamp head is judged on its own chroma and bezel contrast).

Known limits, stated rather than hidden: hand-built features, thresholds fixed
on the dev crops, no learned weights, and no verified amber example anywhere in
the repository - so no accuracy is claimed for yellow. It abstains when torn.
"""

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def _chroma_maps(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Chromatic index per state on 0..1 float BGR - "excess of this state's colour
    over its competitors":
      red    : R beyond both G and B
      yellow : R and G together beyond B (amber glass *and* cream paint, which is
               exactly why chroma alone is never sufficient)
      green  : G beyond the R/B average, so cyan LEDs (G~B) are still recognised
    """
    b = bgr[:, :, 0].astype(np.float32) / 255.0
    g = bgr[:, :, 1].astype(np.float32) / 255.0
    r = bgr[:, :, 2].astype(np.float32) / 255.0
    amber = np.minimum(r, g) - b
    return {
        # Red must be redder than it is amber: subtracting the amber excess stops
        # an amber lamp (R~G>B) from being reported as red, which the frozen v1.4
        # red index did on the held-out set.
        "red": (r - np.maximum(g, b)) - np.maximum(0.0, amber),
        "yellow": amber,
        "green": g - 0.5 * (r + b),
    }


def _fill_holes(comp: np.ndarray) -> np.ndarray:
    """Fill interior holes so a blown-out white lamp core counts as part of it."""
    contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled.astype(bool)


class TrafficLightClassifier:
    """Classifies a traffic-light crop into red / yellow / green / unknown."""

    capability = "heuristic-baseline (per-state chroma components + bezel contrast + reject option)"

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

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _normalise_scale(crop_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Work at a bounded short side so ring radii mean the same thing for a 9x18
        sliver and a 260x520 box. Only modest upscaling, always allow downscale.
        """
        h, w = crop_bgr.shape[:2]
        short = min(h, w)
        if short == 0:
            return crop_bgr, 1.0
        scale = min(config.TL_WORK_SHORT_SIDE / float(short), config.TL_MAX_UPSCALE)
        if abs(scale - 1.0) < 0.08:
            return crop_bgr, 1.0
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(
            crop_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=interp,
        ), scale

    # ------------------------------------------------------------ candidate
    def _candidates(self, bgr: np.ndarray) -> list:
        """Every chromatic component of every state, with its bezel statistics."""
        hh, ww = bgr.shape[:2]
        crop_area = float(hh * ww)
        v = cv2.GaussianBlur(bgr.max(axis=2).astype(np.float32) / 255.0, (3, 3), 0)
        maps = _chroma_maps(bgr)
        out = []

        for state, raw in maps.items():
            m = cv2.GaussianBlur(raw, (5, 5), 0)
            mask = cv2.morphologyEx(
                (m > config.TL_LAMP_CHROMA_FLOOR[state]).astype(np.uint8),
                cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
            )
            n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n_lab):
                area = float(stats[i, cv2.CC_STAT_AREA])
                area_frac = area / crop_area
                if area_frac < config.TL_LAMP_MIN_AREA_FRAC or area_frac > 0.95:
                    continue
                comp = labels == i
                chroma = float(m[comp].mean())
                if chroma < config.TL_LAMP_MIN_CHROMA:
                    continue

                filled = _fill_holes(comp)
                filled_area = float(filled.sum())
                # Ring grown outwards by ~30% of the blob radius: the bezel for a
                # lamp, more of the same paint for a housing face.
                radius = max(3, int(round(0.30 * np.sqrt(filled_area))))
                k = 2 * radius + 1
                grown = cv2.dilate(filled.astype(np.uint8),
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
                ring = grown & ~filled
                if ring.sum() < 6:
                    continue
                contrast = float(np.percentile(v[filled], 80) - np.median(v[ring]))
                bezel_dark = float((v[ring] < 0.6 * np.median(v[filled])).mean())

                # Compactness: a lamp is a filled disc (circularity ~0.6-0.9);
                # housing paint is a crescent, strip or frame round a lens
                # (~0.08-0.25). Measured on sir_image2: the three false amber
                # regions scored circ 0.12-0.25 while the lit green lamp in the
                # same box scored 0.73-0.88.
                contours, _ = cv2.findContours(
                    filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter = sum(cv2.arcLength(c, True) for c in contours)
                circularity = (float(4.0 * np.pi * filled_area / (perimeter ** 2))
                               if perimeter > 0 else 0.0)

                out.append({
                    "state": state,
                    "area_frac": area_frac,
                    "chroma": chroma,
                    "contrast": contrast,
                    "bezel_dark": bezel_dark,
                    "circularity": min(1.0, circularity),
                    "bbox": (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                             int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
                    "cx": float(cents[i][0]),
                    "cy": float(cents[i][1]),
                })
        return out

    def _score(self, cand: dict, structural: bool) -> float:
        """
        Chroma x bezel contrast x bezel darkness, with box-filling regions
        discounted in a structural crop because those are the signal body rather
        than a lamp. Amber gets an extra gate: paint and amber glass are
        chromatically identical, so amber needs the lamp-on-dark-bezel signature
        before it is allowed to win.
        """
        if cand["state"] == "yellow":
            if (cand["contrast"] < config.TL_YELLOW_MIN_CONTRAST
                    or cand["bezel_dark"] < config.TL_YELLOW_MIN_BEZEL
                    or cand["area_frac"] > config.TL_YELLOW_MAX_AREA_FRAC):
                return 0.0

        evidence = min(1.0, max(0.0, cand["contrast"]) / config.TL_RING_FULL_CONTRAST)
        if evidence < config.TL_RING_MIN_EVIDENCE:
            return 0.0
        floor = config.TL_BEZEL_FLOOR
        bezel = floor + (1.0 - floor) * cand["bezel_dark"]
        # A disc is a lamp; a crescent or strip is paint. Weighted, not gated: a
        # tiny genuine lamp has an unreliable perimeter, so it is discounted
        # rather than rejected, and any better-shaped same-state candidate in
        # the same crop still wins on score.
        compact = config.TL_COMPACT_FLOOR + (1.0 - config.TL_COMPACT_FLOOR) * min(
            1.0, cand.get("circularity", 1.0) / config.TL_COMPACT_FULL
        )
        score = cand["chroma"] * evidence * bezel * compact
        if structural and cand["area_frac"] > config.TL_BODY_AREA_FRAC:
            score *= config.TL_BODY_PENALTY
        return score

    # -------------------------------------------------------------- classify
    def classify(self, crop_bgr: np.ndarray, key: Optional[object] = None) -> Tuple[str, float]:
        """
        Classify one crop. When `key` is given the result is subject to temporal
        voting (call once per frame with a stable per-signal key).
        """
        state, conf = self._classify_single(crop_bgr)
        if key is not None:
            state, conf = self._vote(key, state, conf)
        return state, conf

    def _classify_single(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        self.last_debug = {}
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return "unknown", 0.0

        h, w = crop_bgr.shape[:2]
        if (h * w) < self.min_crop_area or w < self.min_width or h < self.min_height:
            self.last_debug["reject"] = "crop smaller than configured minimum"
            return "unknown", 0.0

        structural = min(h, w) > config.TL_TIGHT_CROP_SIDE
        bgr, scale = self._normalise_scale(crop_bgr)
        cands = self._candidates(bgr)
        self.last_debug["regime"] = "structural" if structural else "tight"
        self.last_debug["n_candidates"] = len(cands)
        self.last_debug["scale"] = round(scale, 2)
        if not cands:
            self.last_debug["reject"] = "no chromatic component large enough"
            return "unknown", 0.0

        scored = [(self._score(c, structural), c) for c in cands]
        scored = [(s, c) for s, c in scored if s > 0]
        if not scored:
            self.last_debug["reject"] = "no component contrasts with its surround (flat paint, dark lens or background)"
            return "unknown", 0.0

        scored.sort(key=lambda t: -t[0])
        best_score, best = scored[0]
        self.last_debug.update({
            "state": best["state"],
            "score": round(float(best_score), 3),
            "area_frac": round(best["area_frac"], 3),
            "chroma": round(best["chroma"], 2),
            "contrast": round(best["contrast"], 2),
            "bezel_dark": round(best["bezel_dark"], 2),
            "circularity": round(best.get("circularity", 0.0), 2),
        })

        if best_score < config.TL_MIN_LAMP_SCORE:
            self.last_debug["reject"] = f"weak lamp evidence {best_score:.2f}"
            return "unknown", float(best_score)

        # Only a competitor with comparable support counts as ambiguity; a
        # 1%-of-box speck of another hue is noise, not a rival reading.
        runner_up = 0.0
        for score, cand in scored:
            if cand["state"] == best["state"]:
                continue
            if cand["area_frac"] < config.TL_COMPETITOR_MIN_AREA_FRAC:
                continue
            runner_up = max(runner_up, score)
        self.last_debug["runner_up"] = round(float(runner_up), 3)
        if runner_up > 0 and best_score < config.TL_SCORE_MARGIN * runner_up:
            self.last_debug["reject"] = (
                f"ambiguous: {best['state']} {best_score:.2f} vs competing state {runner_up:.2f}"
            )
            return "unknown", float(best_score)

        return best["state"], float(best_score)

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
