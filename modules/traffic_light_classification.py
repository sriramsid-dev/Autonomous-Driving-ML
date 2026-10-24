"""
Module B (Part 2): Traffic Light State Classification
Algorithm: Spatial-Chrominance & HSV Heuristic Baseline Classifier

IMPORTANT ARCHITECTURAL NOTE:
As specified by project requirements, this module implements a deterministic
Spatial-Chrominance / HSV baseline classifier for Version 1. It is explicitly
labeled as a rule-based baseline and NOT a fine-tuned deep learning classifier.
In Version 2, this baseline will be compared against a model fine-tuned on
labeled autonomous driving datasets (e.g. Bosch / LISA / Waymo).
"""

from typing import Dict, Tuple
import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class TrafficLightClassifier:
    """
    Classifies cropped traffic light images into one of 4 states:
    - 'red'
    - 'yellow'
    - 'green'
    - 'unknown'

    Combines HSV color saturation/value thresholding with spatial layout
    priors (vertical lamp positioning: Red top, Yellow middle, Green bottom).
    Returns 'unknown' for blurry, low-resolution, or ambiguous crops.
    """

    def __init__(
        self,
        min_crop_area: int = config.TL_MIN_CROP_AREA,
        min_width: int = config.TL_MIN_WIDTH,
        min_height: int = config.TL_MIN_HEIGHT,
        ambiguity_threshold: float = config.TL_AMBIGUITY_THRESHOLD,
    ):
        self.min_crop_area = min_crop_area
        self.min_width = min_width
        self.min_height = min_height
        self.ambiguity_threshold = ambiguity_threshold

        # Visual badge colors for each state (BGR format)
        self.state_colors = {
            "red": (0, 0, 255),         # Vivid Red
            "yellow": (0, 215, 255),    # Amber / Gold
            "green": (0, 255, 0),       # Vivid Green
            "unknown": (160, 160, 160), # Neutral Gray
        }

    def classify(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Classifies the state of a cropped traffic light image.
        Returns (predicted_state, state_confidence).
        """
        # Guard clause 1: Null or empty crop
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown", 0.0

        h, w = crop_bgr.shape[:2]

        # Guard clause 2: Sub-resolution / tiny / distant lights
        if (h * w) < self.min_crop_area or w < self.min_width or h < self.min_height:
            return "unknown", 0.0

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # 1. Color Segmentation Masks in HSV Space
        # Red spans 0-12 and 158-180 in OpenCV 8-bit hue
        mask_r1 = cv2.inRange(hsv, np.array([0, 75, 75]), np.array([12, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([158, 75, 75]), np.array([180, 255, 255]))
        mask_red = mask_r1 | mask_r2

        # Yellow / Amber spans 14-38
        mask_yellow = cv2.inRange(hsv, np.array([14, 75, 75]), np.array([38, 255, 255]))

        # Green spans 40-95
        mask_green = cv2.inRange(hsv, np.array([40, 75, 75]), np.array([95, 255, 255]))

        # 2. Spatial Weighting Priors
        # Check orientation: vertical casing vs horizontal casing
        is_horizontal = w > (1.4 * h)
        if not is_horizontal:
            # Vertical: Red top (0.0-0.35), Yellow mid (0.35-0.65), Green bottom (0.65-1.0)
            y_coords = np.linspace(0.0, 1.0, h)[:, None]
            w_red = np.clip(1.6 - 2.2 * y_coords, 0.1, 1.6)
            w_yel = np.clip(1.5 - 4.0 * np.abs(y_coords - 0.5), 0.1, 1.5)
            w_grn = np.clip(2.2 * y_coords - 0.6, 0.1, 1.6)
        else:
            # Horizontal: Red left (0.0-0.35), Yellow mid, Green right (0.65-1.0)
            x_coords = np.linspace(0.0, 1.0, w)[None, :]
            w_red = np.clip(1.6 - 2.2 * x_coords, 0.1, 1.6)
            w_yel = np.clip(1.5 - 4.0 * np.abs(x_coords - 0.5), 0.1, 1.5)
            w_grn = np.clip(2.2 * x_coords - 0.6, 0.1, 1.6)

        # 3. Calculate spatial-weighted activation scores
        score_red = float(np.sum((mask_red > 0) * w_red))
        score_yellow = float(np.sum((mask_yellow > 0) * w_yel))
        score_green = float(np.sum((mask_green > 0) * w_grn))

        total_activation = score_red + score_yellow + score_green

        # Minimum required colored pixel activation (at least 1% of area or 12 px)
        min_activation = max(12.0, 0.008 * (h * w))
        scores = {
            "red": score_red,
            "yellow": score_yellow,
            "green": score_green,
        }
        best_state, best_score = max(scores.items(), key=lambda item: item[1])

        # Guard clause 3: Low luminous / unlit lamp
        if best_score < min_activation or total_activation < 1e-4:
            return "unknown", 0.0

        # Confidence is dominant color ratio
        confidence = best_score / total_activation

        # Guard clause 4: Ambiguous / competing illumination (e.g. sunset or reflections)
        if confidence < self.ambiguity_threshold:
            return "unknown", float(confidence)

        return best_state, float(confidence)

    def annotate(
        self,
        frame_bgr: np.ndarray,
        box: Tuple[float, float, float, float],
        state: str,
        conf: float,
    ) -> np.ndarray:
        """
        Draws traffic light bounding box and a distinct state badge on the frame.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        color = self.state_colors.get(state, self.state_colors["unknown"])

        # Bounding box around traffic light
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # State pill badge
        label = f"TL: {state.upper()} ({conf:.2f})"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        badge_y1 = max(0, y1 - th - baseline - 4)
        badge_y2 = y1
        cv2.rectangle(frame_bgr, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)

        # Text inside badge
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
