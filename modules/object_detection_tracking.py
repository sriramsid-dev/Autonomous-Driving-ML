"""
Module A: Object Detection and Multi-Object Tracking
Algorithm: Pretrained YOLO11n Detector + ByteTrack Multi-Object Tracker
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class ObjectDetectionTracker:
    """
    Performs object detection using pretrained YOLO11n and tracks objects across
    consecutive video frames using ByteTrack.
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        conf_thresh: float = config.DETECTION_CONF_THRESHOLD,
        iou_thresh: float = config.DETECTION_IOU_THRESHOLD,
        tracker_config: str = config.TRACKER_CONFIG,
        target_classes: Optional[Dict[int, str]] = None,
        device: str = config.DEVICE,
        imgsz: int = config.DETECTION_IMGSZ,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.tracker_config = tracker_config
        self.target_classes = target_classes or config.PERCEPTION_TARGET_CLASSES
        self.device = device
        self.imgsz = imgsz
        # Raw (unfiltered) boxes of the most recent call, so the traffic-light
        # module can reuse the same forward pass instead of running the same
        # network a second time on the same frame.
        self.last_raw_boxes: List[Dict] = []

        print(f"[Module A] Initializing ObjectDetectionTracker with model: {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)
        
        # Color palette for classes (BGR)
        self.class_colors = {
            "car": (0, 220, 0),         # Green
            "truck": (255, 140, 0),     # Dark Orange
            "bus": (255, 191, 0),       # Amber / Deep Sky
            "pedestrian": (255, 50, 50),# Blue
            "cyclist": (0, 255, 255),   # Yellow
            "motorcycle": (200, 0, 255),# Magenta
        }
        self.default_color = (0, 255, 128)

        # Track history for motion trails: {track_id: list of (x_center, y_center)}
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.max_history_len = 30

    def reset_tracker(self):
        """
        Reset internal tracking state for a new video sequence.

        V1-FIX: clearing the trail history alone did not reset ByteTrack - the
        association state lives in Ultralytics' predictor and kept producing
        continuing IDs (measured: IDs resumed at 4..10 instead of 1).
        """
        self.track_history.clear()
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            trackers = getattr(predictor, "trackers", None)
            for trk in trackers or []:
                try:
                    trk.reset()
                except Exception:  # noqa: BLE001 - best effort across versions
                    pass
            try:
                predictor.trackers = []
            except Exception:  # noqa: BLE001
                pass
        # Drop the predictor entirely so the next track() call rebuilds it.
        self.model.predictor = None

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Run raw object detection on a single image without temporal tracking.
        Returns list of detection dictionaries.
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            classes=sorted(self.target_classes),
            device=self.device,
            verbose=False,
        )[0]
        dt = (time.perf_counter() - t0) * 1000.0

        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clses = results.boxes.cls.int().cpu().numpy()

            for box, conf, cls_id in zip(boxes, confs, clses):
                if int(cls_id) in self.target_classes:
                    label = self.target_classes[int(cls_id)]
                    detections.append({
                        "box": [float(b) for b in box],
                        "class_id": int(cls_id),
                        "label": label,
                        "confidence": float(conf),
                        "track_id": None,
                    })

        return detections

    def track(self, frame_bgr: np.ndarray, persist: bool = True) -> Tuple[List[Dict], float]:
        """
        Run detection and ByteTrack multi-object tracking on a video frame.
        Maintains unique track IDs across consecutive frames.
        Returns (tracked_objects, inference_time_ms).
        """
        t0 = time.perf_counter()
        # V1-FIX: restrict the tracker itself to the classes we care about.
        # Previously all 80 COCO classes were tracked and filtered afterwards,
        # so signs/benches/lights consumed association slots and track IDs.
        track_classes = sorted(set(self.target_classes) | {config.TRAFFIC_LIGHT_CLASS_ID})
        results = self.model.track(
            frame_bgr,
            conf=min(self.conf_thresh, config.TL_DETECTION_CONF_THRESHOLD),
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            tracker=self.tracker_config,
            persist=persist,
            classes=track_classes,
            device=self.device,
            verbose=False,
        )[0]
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        tracked_objects = []
        self.last_raw_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clses = results.boxes.cls.int().cpu().numpy()
            track_ids = (
                results.boxes.id.int().cpu().numpy()
                if results.boxes.id is not None
                else [None] * len(boxes)
            )

            for box, conf, cls_id, tid in zip(boxes, confs, clses, track_ids):
                self.last_raw_boxes.append({
                    "box": [float(b) for b in box],
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                    "track_id": int(tid) if tid is not None else None,
                })
                if int(cls_id) in self.target_classes:
                    label = self.target_classes[int(cls_id)]
                    t_id = int(tid) if tid is not None else None

                    # Update track history trail
                    if t_id is not None:
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        if t_id not in self.track_history:
                            self.track_history[t_id] = []
                        self.track_history[t_id].append((cx, cy))
                        if len(self.track_history[t_id]) > self.max_history_len:
                            self.track_history[t_id].pop(0)

                    tracked_objects.append({
                        "box": [float(b) for b in box],
                        "class_id": int(cls_id),
                        "label": label,
                        "confidence": float(conf),
                        "track_id": t_id,
                    })

        return tracked_objects, inference_time_ms

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        tracked_objects: List[Dict],
        draw_trails: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, confidence scores, and tracking IDs on the frame.
        """
        annotated = frame_bgr.copy()

        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj["box"]]
            label = obj["label"]
            conf = obj["confidence"]
            tid = obj["track_id"]
            color = self.class_colors.get(label, self.default_color)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Text label
            id_str = f"#{tid} " if tid is not None else ""
            caption = f"{id_str}{label} {conf:.2f}"

            # Label banner
            (tw, th), baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            banner_y1 = max(0, y1 - th - baseline - 4)
            banner_y2 = y1
            cv2.rectangle(
                annotated,
                (x1, banner_y1),
                (x1 + tw + 6, banner_y2),
                color,
                -1,
            )
            # Text inside banner (black or white for readability)
            cv2.putText(
                annotated,
                caption,
                (x1 + 3, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Draw motion trail
            if draw_trails and tid is not None and tid in self.track_history:
                pts = self.track_history[tid]
                if len(pts) > 1:
                    for i in range(1, len(pts)):
                        thickness = int(np.sqrt(float(i + 1)) * 1.2)
                        cv2.line(annotated, pts[i - 1], pts[i], color, thickness)

        return annotated
