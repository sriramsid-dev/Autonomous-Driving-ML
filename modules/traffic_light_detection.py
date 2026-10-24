"""
Module B (Part 1): Traffic Light Detection
Algorithm: Pretrained YOLO11n Detector for COCO Class 9 (Traffic Light)

V1-FIX:
  * input size is configurable (small distant signals were lost when a 5862x3114
    frame was letterboxed to 640 px; measured 2 -> 8 detections at 1280 px);
  * overlapping boxes describing the same signal are merged;
  * `detect_from_boxes()` lets the pipeline reuse the object detector's forward
    pass instead of running a second identical network on the same frame.
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TrafficLightDetector:
    """Detects traffic lights and extracts crops for state classification."""

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        conf_thresh: float = config.TL_DETECTION_CONF_THRESHOLD,
        target_class_id: int = config.TRAFFIC_LIGHT_CLASS_ID,
        device: str = config.DEVICE,
        imgsz: int = config.TL_DETECTION_IMGSZ,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.target_class_id = target_class_id
        self.device = device
        self.imgsz = imgsz

        print(f"[Module B - Detector] Initializing TrafficLightDetector with model: {self.model_path} (imgsz={imgsz})")
        self.model = YOLO(self.model_path)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        """Standalone detection. Returns (detections, inference_time_ms)."""
        t0 = time.perf_counter()
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_thresh,
            classes=[self.target_class_id],
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict] = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            order = np.argsort(-confs)
            kept: List[Sequence[float]] = []
            for idx in order:
                box = boxes[idx]
                if any(_iou(box, k) > config.TL_DUPLICATE_IOU for k in kept):
                    continue  # duplicate description of the same signal
                kept.append(box)
                detections.append({"box": [float(v) for v in box], "confidence": float(confs[idx])})
        return self._finalise(detections, frame_bgr), inference_time_ms

    def detect_from_boxes(self, frame_bgr: np.ndarray, raw_boxes: Sequence[Dict]) -> List[Dict]:
        """
        Reuse a shared detection pass: filter class-9 boxes, merge duplicates and
        crop. `raw_boxes` items need 'box', 'confidence' and 'class_id'.
        """
        detections = [
            {"box": list(b["box"]), "confidence": float(b["confidence"])}
            for b in raw_boxes
            if int(b.get("class_id", -1)) == self.target_class_id
            and float(b.get("confidence", 0.0)) >= self.conf_thresh
        ]
        detections.sort(key=lambda d: -d["confidence"])
        kept: List[Sequence[float]] = []
        merged: List[Dict] = []
        for det in detections:
            if any(_iou(det["box"], k) > config.TL_DUPLICATE_IOU for k in kept):
                continue
            kept.append(det["box"])
            merged.append(det)
        return self._finalise(merged, frame_bgr)

    @staticmethod
    def _finalise(detections: List[Dict], frame_bgr: np.ndarray) -> List[Dict]:
        h, w = frame_bgr.shape[:2]
        out = []
        for det in detections:
            x1 = max(0, int(det["box"][0]))
            y1 = max(0, int(det["box"][1]))
            x2 = min(w, int(det["box"][2]))
            y2 = min(h, int(det["box"][3]))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(det["confidence"]),
                    "crop": frame_bgr[y1:y2, x1:x2].copy(),
                }
            )
        return out
