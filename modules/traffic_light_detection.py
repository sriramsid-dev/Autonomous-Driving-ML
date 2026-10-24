"""
Module B (Part 1): Traffic Light Detection
Algorithm: Pretrained YOLO11n Detector for COCO Class 9 (Traffic Light)
"""

import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class TrafficLightDetector:
    """
    Detects traffic lights in road images and video frames using pretrained YOLO11n.
    Extracts bounding boxes, confidence scores, and cropped patches.
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        conf_thresh: float = config.TL_DETECTION_CONF_THRESHOLD,
        target_class_id: int = config.TRAFFIC_LIGHT_CLASS_ID,
        device: str = config.DEVICE,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.target_class_id = target_class_id
        self.device = device

        print(f"[Module B - Detector] Initializing TrafficLightDetector with model: {self.model_path}")
        self.model = YOLO(self.model_path)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Detect traffic lights in the given image.
        Returns (list_of_detections, inference_time_ms).
        Each detection dict contains:
            - 'box': [x1, y1, x2, y2]
            - 'confidence': float
            - 'crop': cropped np.ndarray (BGR)
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_thresh,
            classes=[self.target_class_id],
            device=self.device,
            verbose=False,
        )[0]
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        h, w = frame_bgr.shape[:2]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(w, int(box[2]))
                y2 = min(h, int(box[3]))

                if x2 > x1 and y2 > y1:
                    crop = frame_bgr[y1:y2, x1:x2].copy()
                else:
                    crop = np.zeros((0, 0, 3), dtype=np.uint8)

                detections.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "crop": crop,
                })

        return detections, inference_time_ms
