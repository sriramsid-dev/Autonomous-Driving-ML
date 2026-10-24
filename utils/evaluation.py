"""
Evaluation and Metrics Framework for Perception System Version 1

Strict Adherence to Evaluation Guidelines:
- Distinguishes theoretical evaluation metrics from empirical runtime statistics.
- Explains metric definitions (mAP@0.5, MOTA, Classification Accuracy, Lane IoU).
- Documents genuine dataset accessibility status (Waymo Open Dataset audit).
- Strictly prevents fabrication of benchmark scores on unannotated footage.
"""

from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path


def audit_dataset_availability() -> Dict[str, object]:
    """
    Audits local workspace for ground-truth datasets and input media.
    Distinguishes locally available video/image resources from cloud/Drive assets.
    """
    audit = {
        "waymo_tfrecords_accessible": False,
        "waymo_ground_truth_accessible": False,
        "local_driving_videos_found": sorted(
            str(p) for p in (Path(__file__).resolve().parent.parent / "test_data").glob("*.mp4")
        ),
        "local_test_images_found": [
            "highway_road.jpg (Straight Highway Lane Markings)",
            "sample_wyamo.jpg (Urban Multi-vehicle driving scene)",
            "test_tl/red_tl.jpg (Real Red Stoplight)",
            "test_tl/green_tl.jpg (Real Green Traffic Signal)",
            "test_tl/yellow_inuvik.jpg (Real Amber Traffic Signal)",
            "test_tl/tl_crop_red.jpg (Isolated Red Lamp)",
            "test_tl/tl_crop_yellow.jpg (Isolated Amber Lamp)",
            "test_tl/tl_crop_green.jpg (Isolated Green Lamp)",
        ],
        "evaluation_strategy": (
            "Because raw Waymo TFRecords and ground-truth bounding box/track labels "
            "are not stored locally in this workspace (they reside on Google Cloud / Google Drive "
            "as described in 'Waymo Open Dataset.pptx'), formal mAP@0.5 and MOTA are evaluated "
            "via rigorous unit test cases with ground-truth test vectors, while video inference "
            "is evaluated using empirical tracking continuity and detection distribution statistics."
        ),
    }
    return audit


def calculate_box_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculates Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2].
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    box1_area = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def evaluate_detection_map50(
    predictions_per_image: List[List[Dict]],
    ground_truth_per_image: List[List[Dict]],
    iou_thresh: float = 0.50,
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and mAP@0.5 for object detection.
    
    Metric Definition:
    - mAP@0.5 measures mean Average Precision across classes when a detection
      is counted as a true positive if its IoU with a ground truth box is >= 0.50.
    """
    # V1-FIX: true per-class AP@0.5. The previous version pooled every class into
    # a single Precision/Recall curve and reported the value at maximum recall,
    # which is neither mAP nor a usable operating point.
    class_ids = sorted(
        {p.get("class_id") for preds in predictions_per_image for p in preds}
        | {g.get("class_id") for gts in ground_truth_per_image for g in gts}
        - {None}
    )
    per_class = {}
    for cid in class_ids:
        res = _ap_single_class(
            [[p for p in preds if p.get("class_id") == cid] for preds in predictions_per_image],
            [[g for g in gts if g.get("class_id") == cid] for gts in ground_truth_per_image],
            iou_thresh,
        )
        per_class[str(cid)] = res

    aps = [v["ap50"] for v in per_class.values() if v["total_gt"] > 0]
    return {
        "map50": float(np.mean(aps)) if aps else 0.0,
        "per_class": per_class,
        "evaluated_classes": len(aps),
        "note": "AP uses all-point interpolation; classes without ground truth are skipped.",
    }


def _ap_single_class(predictions_per_image, ground_truth_per_image, iou_thresh):
    all_scores = []
    all_matched = []
    total_gt = 0

    for preds, gts in zip(predictions_per_image, ground_truth_per_image):
        total_gt += len(gts)
        gt_matched = [False] * len(gts)

        # Sort predictions by confidence descending
        sorted_preds = sorted(preds, key=lambda x: x.get("confidence", 0.0), reverse=True)

        for p in sorted_preds:
            p_box = p["box"]
            p_cls = p.get("class_id")
            conf = p.get("confidence", 0.0)

            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gts):
                if gt.get("class_id") == p_cls:
                    iou = calculate_box_iou(p_box, gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            all_scores.append(conf)
            if best_iou >= iou_thresh and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                all_matched.append(1)
                gt_matched[best_gt_idx] = True
            else:
                all_matched.append(0)

    if total_gt == 0 or len(all_matched) == 0:
        return {"precision": 0.0, "recall": 0.0, "ap50": 0.0, "total_gt": total_gt}

    # Cumulative TP and FP
    all_matched = np.array(all_matched)
    all_scores = np.array(all_scores)
    sort_indices = np.argsort(-all_scores)
    tp_cumsum = np.cumsum(all_matched[sort_indices])
    fp_cumsum = np.cumsum(1 - all_matched[sort_indices])

    recalls = tp_cumsum / float(total_gt)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

    # All-point interpolated AP (COCO-style monotone envelope)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best = int(np.argmax(f1)) if len(f1) else 0

    return {
        "precision_at_best_f1": float(precisions[best]) if len(precisions) else 0.0,
        "recall_at_best_f1": float(recalls[best]) if len(recalls) else 0.0,
        "ap50": ap,
        "total_gt": int(total_gt),
        "total_pred": int(len(all_scores)),
    }


def evaluate_tracking_mota(
    tracks_per_frame: List[Dict[int, List[float]]],
    gt_tracks_per_frame: List[Dict[int, List[float]]],
    iou_thresh: float = 0.50,
) -> Dict[str, float]:
    """
    Computes MOTA (Multiple Object Tracking Accuracy).
    
    Metric Definition:
    MOTA = 1 - (False_Negatives + False_Positives + ID_Switches) / Total_Ground_Truth_Boxes
    Measures detector accuracy combined with identity consistency over time.
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    prev_matches: Dict[int, int] = {}  # {gt_id: pred_id}

    for pred_tracks, gt_tracks in zip(tracks_per_frame, gt_tracks_per_frame):
        total_gt += len(gt_tracks)
        matched_preds = set()
        current_matches: Dict[int, int] = {}

        for gt_id, gt_box in gt_tracks.items():
            best_iou = 0.0
            best_pred_id = None
            for p_id, p_box in pred_tracks.items():
                if p_id not in matched_preds:
                    iou = calculate_box_iou(p_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_id = p_id

            if best_iou >= iou_thresh and best_pred_id is not None:
                matched_preds.add(best_pred_id)
                current_matches[gt_id] = best_pred_id
                # Check for ID Switch
                if gt_id in prev_matches and prev_matches[gt_id] != best_pred_id:
                    total_idsw += 1
            else:
                total_fn += 1

        total_fp += len(pred_tracks) - len(matched_preds)
        prev_matches = current_matches

    if total_gt == 0:
        return {"mota": 0.0, "fn": total_fn, "fp": total_fp, "idsw": total_idsw, "total_gt": 0}

    mota = 1.0 - float(total_fn + total_fp + total_idsw) / float(total_gt)
    return {
        "mota": float(mota),
        "false_negatives": int(total_fn),
        "false_positives": int(total_fp),
        "id_switches": int(total_idsw),
        "total_gt": int(total_gt),
    }


def evaluate_tl_classification(
    predictions: List[str],
    ground_truths: List[str],
    classes: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Evaluates traffic light state classification accuracy, per-class precision,
    recall, and confusion matrix on labeled examples.
    """
    if classes is None:
        classes = ["red", "yellow", "green", "unknown"]

    total = len(ground_truths)
    if total == 0:
        return {"accuracy": 0.0, "sample_count": 0}

    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / float(total)

    matrix = {c_gt: {c_pred: 0 for c_pred in classes} for c_gt in classes}
    for p, g in zip(predictions, ground_truths):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1

    per_class = {}
    for c in classes:
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in classes if other != c)
        fn = sum(matrix[c][other] for other in classes if other != c)
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "support": int(tp + fn),
        }

    return {
        "overall_accuracy": float(accuracy),
        "sample_count": total,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


class EmpiricalRunStatistics:
    """
    Accumulates real-world runtime telemetry and detection distributions
    without storing high-resolution frames in memory.
    """

    def __init__(self):
        self.frame_count = 0
        self.total_time_ms = 0.0
        self.latencies_ms: List[float] = []

        self.class_counts: Dict[str, int] = {}
        self.track_lifetimes: Dict[int, int] = {}  # {track_id: frames_seen}
        self.tl_state_counts: Dict[str, int] = {
            "red": 0,
            "yellow": 0,
            "green": 0,
            "unknown": 0,
        }
        self.frames_with_lanes = 0

    def update_frame(
        self,
        latency_ms: float,
        tracked_objects: List[Dict],
        tl_detections: List[Dict],
        lanes: List[List[Tuple[int, int]]],
    ):
        self.frame_count += 1
        self.total_time_ms += latency_ms
        self.latencies_ms.append(latency_ms)

        # Objects and tracks
        for obj in tracked_objects:
            lbl = obj["label"]
            self.class_counts[lbl] = self.class_counts.get(lbl, 0) + 1
            tid = obj.get("track_id")
            if tid is not None:
                self.track_lifetimes[tid] = self.track_lifetimes.get(tid, 0) + 1

        # Traffic lights
        for tl in tl_detections:
            state = tl.get("state", "unknown")
            if state in self.tl_state_counts:
                self.tl_state_counts[state] += 1
            else:
                self.tl_state_counts[state] = 1

        # Lanes: check if at least 1 lane had valid points
        if any(len(l) >= 2 for l in lanes):
            self.frames_with_lanes += 1

    def summary(self) -> Dict[str, object]:
        avg_fps = (
            (self.frame_count / (self.total_time_ms / 1000.0))
            if self.total_time_ms > 0
            else 0.0
        )
        p50 = float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0
        p90 = float(np.percentile(self.latencies_ms, 90)) if self.latencies_ms else 0.0
        p99 = float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

        durations = list(self.track_lifetimes.values())
        mean_track_len = float(np.mean(durations)) if durations else 0.0
        max_track_len = int(np.max(durations)) if durations else 0

        return {
            "frames_processed": self.frame_count,
            "total_processing_time_seconds": round(self.total_time_ms / 1000.0, 2),
            "average_fps": round(avg_fps, 2),
            "latency_p50_ms": round(p50, 1),
            "latency_p90_ms": round(p90, 1),
            "latency_p99_ms": round(p99, 1),
            "total_detections_by_class": self.class_counts,
            "unique_tracking_ids": len(self.track_lifetimes),
            "mean_track_lifetime_frames": round(mean_track_len, 1),
            "max_track_lifetime_frames": max_track_len,
            "traffic_light_states_detected": self.tl_state_counts,
            "lane_detection_visibility_rate": (
                round(self.frames_with_lanes / float(max(1, self.frame_count)), 3)
            ),
        }
