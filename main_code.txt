"""
Autonomous Driving Perception System - Version 1 (Pretrained Baseline)
Main Integrated Pipeline & Execution CLI

Integrates:
1. Module A: Object Detection and Multi-Object Tracking (YOLO11n + ByteTrack)
2. Module B: Traffic Light Detection and State Classification (YOLO11n + Spatial-Chrominance Baseline)
3. Module C: Lane Detection (Ultra-Fast Lane Detection - TuSimple ResNet18)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from modules.object_detection_tracking import ObjectDetectionTracker
from modules.traffic_light_detection import TrafficLightDetector
from modules.traffic_light_classification import TrafficLightClassifier
from modules.lane_detection import UltraFastLaneDetector
from utils.video_utils import (
    create_video_writer,
    draw_perception_hud,
    get_video_properties,
    save_annotated_frame,
)
from utils.evaluation import (
    EmpiricalRunStatistics,
    audit_dataset_availability,
    evaluate_tl_classification,
)


def parse_module_selection(modules_str: str) -> Tuple[bool, bool, bool]:
    """
    Parses modules string into (enable_objects, enable_tl, enable_lanes).
    Accepts 'all', or comma-separated values like 'A,B,C', 'objects,lanes', 'traffic_light', etc.
    """
    if not modules_str or modules_str.strip().lower() == "all":
        return True, True, True

    tokens = [t.strip().lower() for t in modules_str.split(",")]
    run_a = any(t in ["a", "object", "objects", "tracking", "obj"] for t in tokens)
    run_b = any(t in ["b", "tl", "traffic_light", "traffic_lights", "traffic-light", "lights"] for t in tokens)
    run_c = any(t in ["c", "lane", "lanes", "ufld"] for t in tokens)

    return run_a, run_b, run_c


class PerceptionPipeline:
    """
    End-to-end perception engine combining Object Tracking, Traffic Light
    State Classification, and Ultra-Fast Lane Detection.
    """

    def __init__(
        self,
        enable_objects: bool = True,
        enable_traffic_lights: bool = True,
        enable_lanes: bool = True,
        device: str = config.DEVICE,
    ):
        print(f"\n============================================================")
        print(f" Initializing Autonomous Driving Perception Pipeline v1")
        print(f" Target Device: {device}")
        print(f"============================================================")

        self.enable_objects = enable_objects
        self.enable_traffic_lights = enable_traffic_lights
        self.enable_lanes = enable_lanes

        # Module A
        if self.enable_objects:
            self.tracker = ObjectDetectionTracker(device=device)
        else:
            self.tracker = None
            print("[Module A] Disabled.")

        # Module B
        if self.enable_traffic_lights:
            self.tl_detector = TrafficLightDetector(device=device)
            self.tl_classifier = TrafficLightClassifier()
        else:
            self.tl_detector = None
            self.tl_classifier = None
            print("[Module B] Disabled.")

        # Module C
        if self.enable_lanes:
            self.lane_detector = UltraFastLaneDetector(device=device)
        else:
            self.lane_detector = None
            print("[Module C] Disabled.")

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
        total_frames: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        Runs perception modules on a single frame and renders visual annotations + HUD.
        Returns (annotated_frame, frame_telemetry).
        """
        t_start = time.perf_counter()
        annotated = frame_bgr.copy()

        # 1. Module A: Object Detection & Tracking (Run first to obtain occlusion boxes)
        tracked_objects = []
        vehicles_count = 0
        pedestrians_count = 0
        vehicle_boxes = []
        all_object_boxes = []
        if self.tracker is not None:
            tracked_objects, _ = self.tracker.track(frame_bgr, persist=True)
            for obj in tracked_objects:
                all_object_boxes.append(obj["box"])
                if obj["label"] in ["car", "truck", "bus"]:
                    vehicles_count += 1
                    vehicle_boxes.append(obj["box"])
                elif obj["label"] in ["pedestrian", "cyclist"]:
                    pedestrians_count += 1

        # 2. Module C: Lane Detection (Draw on clean background, with vehicle occlusion filtering)
        lanes = []
        lane_status = "Inactive"
        if self.lane_detector is not None and self.lane_detector.is_loaded:
            lanes, _ = self.lane_detector.detect(frame_bgr, obstacle_boxes=all_object_boxes)
            annotated = self.lane_detector.annotate_frame(annotated, lanes, draw_drive_area=True)
            active_lane_count = sum(1 for l in lanes if len(l) >= 2)
            if active_lane_count >= 2 and self.lane_detector.last_corridor_established:
                lane_status = f"{active_lane_count} Lanes (Corridor Active)"
            elif active_lane_count > 0:
                lane_status = f"{active_lane_count} Lanes (No Corridor)"
            elif vehicle_boxes:
                lane_status = "No Lanes (Vehicle Occluded)"
            else:
                lane_status = "Insufficient Confidence"

        # Annotate tracked objects on top of lanes
        if self.tracker is not None:
            annotated = self.tracker.annotate_frame(annotated, tracked_objects, draw_trails=True)

        # 3. Module B: Traffic Light Detection & State Classification
        tl_detections = []
        top_tl_state = "None"
        if self.tl_detector is not None and self.tl_classifier is not None:
            raw_tls, _ = self.tl_detector.detect(frame_bgr)
            for tl in raw_tls:
                crop = tl["crop"]
                state, s_conf = self.tl_classifier.classify(crop)
                tl["state"] = state
                tl["state_conf"] = s_conf
                tl_detections.append(tl)

                # Annotate traffic light badge
                annotated = self.tl_classifier.annotate(
                    annotated, tl["box"], state, s_conf
                )

            # Determine dominant/closest traffic light state for HUD
            if tl_detections:
                # Prioritize non-unknown states with highest confidence
                valid_states = [t for t in tl_detections if t["state"] != "unknown"]
                if valid_states:
                    best_tl = max(valid_states, key=lambda x: x["state_conf"])
                    top_tl_state = best_tl["state"]
                else:
                    top_tl_state = tl_detections[0]["state"]

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        fps_val = (1000.0 / latency_ms) if latency_ms > 0 else 0.0

        stats = {
            "frame_idx": frame_idx,
            "total_frames": total_frames,
            "fps": fps_val,
            "latency_ms": latency_ms,
            "vehicles_count": vehicles_count,
            "pedestrians_count": pedestrians_count,
            "active_tracks": len([o for o in tracked_objects if o["track_id"] is not None]),
            "tl_state": top_tl_state,
            "lane_status": lane_status,
        }

        # Render top HUD banner
        annotated = draw_perception_hud(annotated, stats)

        telemetry = {
            "stats": stats,
            "tracked_objects": tracked_objects,
            "tl_detections": tl_detections,
            "lanes": lanes,
        }
        return annotated, telemetry

    def process_image(
        self,
        image_input: object,
        output_image_path: Optional[str] = None,
        modules: Optional[str] = "all",
        is_crop: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        Runs perception inference on a single static image.
        Supports Module A (Objects), Module B (Traffic Lights), and Module C (Lanes).
        Saves annotated image with bounding boxes, labels, initial tracking IDs,
        lane curves, traffic light badges, and telemetry HUD.
        """
        if isinstance(image_input, (str, Path)):
            image_path_str = str(image_input)
            frame_bgr = cv2.imread(image_path_str)
            if frame_bgr is None:
                raise FileNotFoundError(f"Failed to load image from: {image_path_str}")
            stem = Path(image_path_str).stem
        elif isinstance(image_input, np.ndarray):
            frame_bgr = image_input
            image_path_str = "in_memory_image.jpg"
            stem = "in_memory_image"
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        h, w = frame_bgr.shape[:2]
        if output_image_path is None:
            output_image_path = str(config.RESULT_IMAGES_DIR / f"{stem}_annotated.jpg")

        # Parse module selection
        is_crop_file = (
            is_crop
            or ("crop" in Path(image_path_str).name.lower())
            or ("tl_crop" in Path(image_path_str).name.lower())
        )

        effective_modules = modules or "all"
        # If image is an isolated traffic light crop and user did not specify explicit modules,
        # default to Module B only (no lane lines or objects on a cropped lamp)
        if is_crop_file and effective_modules == "all":
            effective_modules = "B"

        req_a, req_b, req_c = parse_module_selection(effective_modules)
        run_a = req_a and (self.tracker is not None)
        run_b = req_b and (self.tl_detector is not None and self.tl_classifier is not None)
        run_c = req_c and (self.lane_detector is not None and self.lane_detector.is_loaded)

        t_start = time.perf_counter()
        annotated = frame_bgr.copy()

        # 1. Module A: Object Detection and Tracking (Run first to obtain occlusion boxes)
        tracked_objects = []
        vehicles_count = 0
        pedestrians_count = 0
        vehicle_boxes = []
        all_object_boxes = []
        obj_time_ms = 0.0
        if run_a:
            tracked_objects, obj_time_ms = self.tracker.track(frame_bgr, persist=False)
            for obj in tracked_objects:
                all_object_boxes.append(obj["box"])
                if obj["label"] in ["car", "truck", "bus"]:
                    vehicles_count += 1
                    vehicle_boxes.append(obj["box"])
                elif obj["label"] in ["pedestrian", "cyclist"]:
                    pedestrians_count += 1

        # 2. Module C: Lane Detection (Draw on clean background, with vehicle occlusion filtering)
        lanes = []
        lane_status = "Inactive"
        active_lane_count = 0
        lane_time_ms = 0.0
        corridor_active = False
        if run_c:
            lanes, lane_time_ms = self.lane_detector.detect(frame_bgr, obstacle_boxes=all_object_boxes)
            annotated = self.lane_detector.annotate_frame(annotated, lanes, draw_drive_area=True)
            active_lane_count = sum(1 for l in lanes if len(l) >= 2)
            corridor_active = self.lane_detector.last_corridor_established
            if active_lane_count >= 2 and corridor_active:
                lane_status = f"{active_lane_count} Lanes (Corridor Active)"
            elif active_lane_count > 0:
                lane_status = f"{active_lane_count} Lanes (No Corridor)"
            elif vehicle_boxes:
                lane_status = "No Lanes (Vehicle Occluded)"
            else:
                lane_status = "Insufficient Confidence"

        # Annotate objects on top of lanes
        if run_a:
            annotated = self.tracker.annotate_frame(annotated, tracked_objects, draw_trails=False)

        # 3. Module B: Traffic Light Detection and Classification
        tl_detections = []
        top_tl_state = "None"
        tl_time_ms = 0.0
        if run_b:
            if is_crop_file:
                # Direct crop classification on pre-cropped traffic signal
                direct_state, direct_conf = self.tl_classifier.classify(frame_bgr)
                crop_box = [4.0, 4.0, float(w - 4), float(h - 4)]
                tl_detections.append({
                    "box": crop_box,
                    "confidence": float(direct_conf),
                    "crop": frame_bgr,
                    "state": direct_state,
                    "state_conf": float(direct_conf),
                    "is_direct_crop": True,
                })
                annotated = self.tl_classifier.annotate(
                    annotated, crop_box, direct_state, direct_conf
                )
                top_tl_state = direct_state
            else:
                # Full scene: detect traffic lights with YOLO then classify each crop
                raw_tls, tl_time_ms = self.tl_detector.detect(frame_bgr)
                for tl in raw_tls:
                    crop = tl["crop"]
                    state, s_conf = self.tl_classifier.classify(crop)
                    tl["state"] = state
                    tl["state_conf"] = s_conf
                    tl_detections.append(tl)
                    annotated = self.tl_classifier.annotate(
                        annotated, tl["box"], state, s_conf
                    )

                if tl_detections:
                    valid_states = [t for t in tl_detections if t["state"] != "unknown"]
                    if valid_states:
                        best_tl = max(valid_states, key=lambda x: x["state_conf"])
                        top_tl_state = best_tl["state"]
                    else:
                        top_tl_state = tl_detections[0]["state"]

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Construct modules execution string
        mod_tags = []
        if run_a:
            mod_tags.append("A")
        if run_b:
            mod_tags.append("B")
        if run_c:
            mod_tags.append("C")
        modules_str = ", ".join(mod_tags) if mod_tags else "None"

        stats = {
            "is_image": True,
            "modules_str": modules_str,
            "fps": (1000.0 / latency_ms) if latency_ms > 0 else 0.0,
            "latency_ms": latency_ms,
            "vehicles_count": vehicles_count,
            "pedestrians_count": pedestrians_count,
            "active_tracks": len([o for o in tracked_objects if o["track_id"] is not None]),
            "tl_state": top_tl_state,
            "lane_status": lane_status,
        }

        # Render top HUD banner
        annotated = draw_perception_hud(annotated, stats)

        # Save annotated image
        save_annotated_frame(annotated, output_image_path)

        telemetry = {
            "image_path": str(image_path_str),
            "output_path": str(output_image_path),
            "resolution": f"{w}x{h}",
            "latency_ms": round(latency_ms, 2),
            "fps_equivalent": round(1000.0 / latency_ms, 2) if latency_ms > 0 else 0.0,
            "modules_executed": {
                "module_A_object_tracking": run_a,
                "module_B_traffic_lights": run_b,
                "module_C_lane_detection": run_c,
            },
            "detection_counts": {
                "vehicles": vehicles_count,
                "pedestrians": pedestrians_count,
                "total_objects": len(tracked_objects),
                "traffic_lights": len(tl_detections),
                "active_lanes": active_lane_count,
            },
            "module_details": {
                "module_A": {
                    "tracked_objects": [
                        {
                            "label": obj["label"],
                            "confidence": round(obj["confidence"], 3),
                            "box": [round(b, 1) for b in obj["box"]],
                            "track_id": obj["track_id"],
                        }
                        for obj in tracked_objects
                    ],
                    "track_id_note": "Initial track IDs assigned by ByteTrack on static image. Multi-frame continuity/MOTA cannot be evaluated on single images.",
                },
                "module_B": {
                    "traffic_lights": [
                        {
                            "box": [round(b, 1) for b in tl["box"]],
                            "detector_confidence": round(tl["confidence"], 3),
                            "classified_state": tl["state"],
                            "state_confidence": round(tl["state_conf"], 3),
                        }
                        for tl in tl_detections
                    ],
                    "dominant_state": top_tl_state,
                },
                "module_C": {
                    "active_lane_count": active_lane_count,
                    "lane_point_counts": [len(l) for l in lanes],
                },
            },
            "evaluation_notes": {
                "accuracy_metrics_disclaimer": "Ground-truth annotations are not available for unannotated static images. In accordance with benchmark rigor, theoretical metrics (mAP@0.5, MOTA, precision, recall) are NOT fabricated. Detection counts and processing times are reported empirically.",
                "tracking_disclaimer": "Tracking IDs reflect single-frame tracker initialization. Temporal tracking accuracy, persistence, and MOTA require sequential video frames and cannot be claimed from a single static image.",
            },
        }

        return annotated, telemetry

    def process_video(
        self,
        video_path: str,
        output_video_path: str,
        max_frames: Optional[int] = None,
        save_keyframes: bool = True,
        keyframe_dir: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Streams input video frame-by-frame, performs perception inference,
        writes annotated video directly to disk (O(1) memory complexity),
        and returns cumulative empirical run statistics.
        """
        print(f"\n[Pipeline] Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video file: {video_path}")

        props = get_video_properties(video_path)
        fps = props["fps"] if props["fps"] > 0 else 25.0
        total_video_frames = props["frame_count"]
        w = props["width"]
        h = props["height"]

        frames_to_process = min(total_video_frames, max_frames) if max_frames else total_video_frames
        print(f"[Pipeline] Video Resolution: {w}x{h} @ {fps:.1f} FPS")
        print(f"[Pipeline] Processing {frames_to_process} frames...")

        # Reset tracking history for clean video start
        if self.tracker:
            self.tracker.reset_tracker()

        writer = create_video_writer(
            output_video_path,
            fps=fps,
            width=w,
            height=h,
            preferred_codec=config.DEFAULT_VIDEO_CODEC,
        )

        run_stats = EmpiricalRunStatistics()
        keyframe_dir_path = Path(keyframe_dir) if keyframe_dir else config.RESULT_IMAGES_DIR
        keyframe_dir_path.mkdir(parents=True, exist_ok=True)

        # Snapshot keyframe indices (evenly spaced)
        keyframe_indices = set(np.linspace(10, frames_to_process - 5, 5, dtype=int).tolist())

        f_idx = 0
        t0_all = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and f_idx >= max_frames):
                    break

                f_idx += 1
                annotated, telem = self.process_frame(
                    frame, frame_idx=f_idx, total_frames=frames_to_process
                )

                # Write directly to disk
                writer.write(annotated)

                # Update running metrics
                run_stats.update_frame(
                    latency_ms=telem["stats"]["latency_ms"],
                    tracked_objects=telem["tracked_objects"],
                    tl_detections=telem["tl_detections"],
                    lanes=telem["lanes"],
                )

                # Save representative keyframes
                if save_keyframes and f_idx in keyframe_indices:
                    snap_name = f"keyframe_frame_{f_idx:04d}.jpg"
                    snap_path = keyframe_dir_path / snap_name
                    cv2.imwrite(str(snap_path), annotated)

                if f_idx % 25 == 0 or f_idx == frames_to_process:
                    print(
                        f"  -> Frame {f_idx}/{frames_to_process} "
                        f"({telem['stats']['fps']:.1f} FPS, "
                        f"Vehicles: {telem['stats']['vehicles_count']}, "
                        f"Tracks: {telem['stats']['active_tracks']})"
                    )

        finally:
            cap.release()
            writer.release()

        total_elapsed = time.perf_counter() - t0_all
        summary = run_stats.summary()
        summary["wall_clock_time_seconds"] = round(total_elapsed, 2)
        summary["output_video_path"] = str(output_video_path)
        summary["video_resolution"] = f"{w}x{h}"

        print(f"\n[Pipeline] Video processing complete!")
        print(f"  Total Frames Processed: {summary['frames_processed']}")
        print(f"  Average Processing Speed: {summary['average_fps']} FPS")
        print(f"  Wall-clock Duration: {summary['wall_clock_time_seconds']} s")
        print(f"  Output saved to: {output_video_path}")

        return summary


def run_unit_tests_and_static_benchmarks(device: str = config.DEVICE) -> Dict[str, object]:
    """
    Executes isolated unit tests on static test assets:
    - Object Detection on driving scene
    - Lane Detection on highway road image
    - Traffic Light Detection & Classification on Red, Yellow, Green, Unknown samples
    """
    print("\n============================================================")
    print(" Running Module Unit Tests on Curated Test Assets")
    print("============================================================")

    results = {}
    test_dir = config.TEST_DATA_DIR
    out_img_dir = config.RESULT_IMAGES_DIR

    # 1. Module A Test: Multi-vehicle detection on driving scene
    sample_img_path = str(test_dir / "sample_wyamo.jpg")
    if os.path.exists(sample_img_path):
        tracker = ObjectDetectionTracker(device=device)
        frame = cv2.imread(sample_img_path)
        dets = tracker.detect(frame)
        ann = tracker.annotate_frame(frame, dets, draw_trails=False)
        out_p = out_img_dir / "test_module_A_objects.jpg"
        cv2.imwrite(str(out_p), ann)
        results["module_A"] = {
            "test_image": "sample_wyamo.jpg",
            "detections_count": len(dets),
            "detected_labels": [d["label"] for d in dets],
            "confidences": [round(d["confidence"], 2) for d in dets],
            "output_saved": str(out_p),
        }
        print(f"[Module A Test] Detected {len(dets)} objects: {results['module_A']['detected_labels']}")

    # 2. Module C Test: Lane Detection on highway road image
    highway_path = str(test_dir / "highway_road.jpg")
    if os.path.exists(highway_path):
        lane_det = UltraFastLaneDetector(device=device)
        hw_frame = cv2.imread(highway_path)
        lanes, l_ms = lane_det.detect(hw_frame)
        ann_hw = lane_det.annotate_frame(hw_frame, lanes, draw_drive_area=True)
        out_hw = out_img_dir / "test_module_C_lanes.jpg"
        cv2.imwrite(str(out_hw), ann_hw)
        lane_lens = [len(l) for l in lanes]
        results["module_C"] = {
            "test_image": "highway_road.jpg",
            "lane_points_detected": lane_lens,
            "active_lanes": sum(1 for pts in lane_lens if pts >= 2),
            "inference_time_ms": round(l_ms, 1),
            "output_saved": str(out_hw),
        }
        print(f"[Module C Test] Detected lanes with points {lane_lens} in {l_ms:.1f} ms")

    # 3. Module B Test: Traffic Light Detection & State Classification on Labeled Crops
    tl_clf = TrafficLightClassifier()
    tl_cases = [
        ("tl_crop_red.jpg", "red"),
        ("tl_crop_yellow.jpg", "yellow"),
        ("tl_crop_green.jpg", "green"),
    ]
    preds = []
    gts = []
    details = []

    for fname, gt_label in tl_cases:
        p = str(test_dir / fname)
        if os.path.exists(p):
            crop = cv2.imread(p)
            pred_state, conf = tl_clf.classify(crop)
            preds.append(pred_state)
            gts.append(gt_label)
            details.append({
                "file": fname,
                "ground_truth": gt_label,
                "predicted": pred_state,
                "confidence": round(conf, 2),
                "is_correct": pred_state == gt_label,
            })

    # Also test scene images with traffic light detector
    tl_detector = TrafficLightDetector(device=device)
    scene_tests = ["red_tl.jpg", "green_tl.jpg", "yellow_inuvik.jpg"]
    scene_results = []
    for sf in scene_tests:
        sp = str(test_dir / sf)
        if os.path.exists(sp):
            s_img = cv2.imread(sp)
            s_dets, _ = tl_detector.detect(s_img)
            s_ann = s_img.copy()
            for d in s_dets:
                st, c = tl_clf.classify(d["crop"])
                d["state"] = st
                d["state_conf"] = c
                s_ann = tl_clf.annotate(s_ann, d["box"], st, c)
            out_s = out_img_dir / f"test_module_B_{sf}"
            cv2.imwrite(str(out_s), s_ann)
            scene_results.append({
                "file": sf,
                "detected_lights_count": len(s_dets),
                "states": [d["state"] for d in s_dets],
            })

    eval_clf = evaluate_tl_classification(preds, gts)
    results["module_B"] = {
        "labeled_crops_tested": details,
        "classification_accuracy": eval_clf["overall_accuracy"],
        "scene_detection_tests": scene_results,
    }
    print(f"[Module B Test] Labeled state classification accuracy: {eval_clf['overall_accuracy'] * 100:.1f}%")

    return results


def run_all_image_tests(device: str = config.DEVICE) -> Dict[str, object]:
    """
    Executes perception testing across all curated local test images.
    Covers Module A (Objects), Module B (Traffic Lights), and Module C (Lanes).
    Generates a structured test summary listing successful tests, failed tests,
    empirical detection counts, latency, and system limitations.
    """
    print("\n" + "=" * 70)
    print(" Running Multi-Module Perception Test Suite on Local Test Images")
    print(f" Target Device: {device}")
    print("=" * 70)

    # Each test case specifies which modules are appropriate for that image.
    # Traffic-light crops: Module B only (no lane/object detection on a lamp close-up).
    # Highway road: Module C (lanes); Module A optional but no expected vehicles in view.
    # Urban vehicle close-up: Module A only (lane detector fires on car body features).
    # Full road scenes: all relevant modules.
    test_cases = [
        {
            "filename": "sample_wyamo.jpg",
            "name": "Urban Vehicle Close-Up (Waymo Jaguar)",
            "modules": "A",
            "visual_verification_target": "Large bounding box covering the dominant vehicle (99% of frame width is expected — vehicle physically fills the image).",
            "expected_tl_state": None,
        },
        {
            "filename": "highway_road.jpg",
            "name": "Highway Road Lane Markings",
            "modules": "C",
            "visual_verification_target": "Two ego-lane boundaries with a bounded green drive corridor between them, not bleeding onto the dirt shoulder.",
            "expected_tl_state": None,
        },
        {
            "filename": "red_tl.jpg",
            "name": "Red Traffic Light Full Scene",
            "modules": "A,B",
            "visual_verification_target": "Traffic light detected and classified as RED.",
            "expected_tl_state": "red",
        },
        {
            "filename": "green_tl.jpg",
            "name": "Green Traffic Light Full Scene",
            "modules": "B",
            "visual_verification_target": "Traffic light detected and classified as GREEN.",
            "expected_tl_state": "green",
        },
        {
            "filename": "yellow_inuvik.jpg",
            "name": "Yellow Traffic Light Full Scene",
            "modules": "A,B",
            "visual_verification_target": "Traffic light detected and classified as YELLOW.",
            "expected_tl_state": "yellow",
        },
        {
            "filename": "tl_crop_red.jpg",
            "name": "Isolated Red Lamp Crop",
            "modules": "B",
            "visual_verification_target": "Direct crop classified as RED.",
            "expected_tl_state": "red",
        },
        {
            "filename": "tl_crop_green.jpg",
            "name": "Isolated Green Lamp Crop",
            "modules": "B",
            "visual_verification_target": "Direct crop classified as GREEN.",
            "expected_tl_state": "green",
        },
        {
            "filename": "tl_crop_yellow.jpg",
            "name": "Isolated Yellow Lamp Crop",
            "modules": "B",
            "visual_verification_target": "Direct crop classified as YELLOW.",
            "expected_tl_state": "yellow",
        },
        {
            "filename": "sample_gemini.jpg",
            "name": "Indoor Group Photo (Object Detection Only)",
            "modules": "A",
            "visual_verification_target": "Pedestrians detected. Not a driving scene — Module C (lanes) intentionally disabled.",
            "expected_tl_state": None,
        },
        {
            "filename": "wyamo_f60.jpg",
            "name": "Driving Frame f60 (All Modules)",
            "modules": "all",
            "visual_verification_target": "Vehicle detection and lane detection on a dashcam driving frame.",
            "expected_tl_state": None,
        },
    ]

    pipeline = PerceptionPipeline(device=device)
    test_results = []
    successful_count = 0
    failed_count = 0

    for tc in test_cases:
        fname = tc["filename"]
        img_path = config.TEST_DATA_DIR / fname
        test_entry = {
            "test_name": tc["name"],
            "filename": fname,
            "modules_requested": tc["modules"],
            "status": "FAIL",
            "error": None,
            "latency_ms": 0.0,
            "detections": {},
            "modules_run": [],
            "output_saved": None,
            "tl_states": [],
            "visual_verification_target": tc.get("visual_verification_target", ""),
            "visual_verification_result": "NOT VERIFIED (requires human inspection of output image)",
        }

        if not img_path.exists():
            test_entry["error"] = f"File not found: {img_path}"
            failed_count += 1
            test_results.append(test_entry)
            print(f" [SKIP] {fname}: File not found")
            continue

        try:
            out_img_name = f"test_result_{fname}"
            out_img_path = str(config.RESULT_IMAGES_DIR / out_img_name)
            _, telem = pipeline.process_image(
                image_input=img_path,
                output_image_path=out_img_path,
                modules=tc["modules"],
            )

            # Record success
            test_entry["status"] = "PASS"
            test_entry["latency_ms"] = telem["latency_ms"]
            test_entry["detections"] = telem["detection_counts"]
            test_entry["modules_run"] = [
                k for k, v in telem["modules_executed"].items() if v
            ]
            test_entry["output_saved"] = telem["output_path"]
            test_entry["tl_states"] = [
                f"{t['classified_state']} ({t['state_confidence']:.2f})"
                for t in telem["module_details"]["module_B"]["traffic_lights"]
            ]
            successful_count += 1

            det_summary_str = (
                f"Vehicles: {telem['detection_counts']['vehicles']}, "
                f"TLs: {telem['detection_counts']['traffic_lights']} {test_entry['tl_states']}, "
                f"Lanes: {telem['detection_counts']['active_lanes']}"
            )
            print(f" [PASS] {fname:20s} | {telem['latency_ms']:6.1f} ms | {det_summary_str}")

        except Exception as e:
            test_entry["status"] = "FAIL"
            test_entry["error"] = str(e)
            failed_count += 1
            print(f" [FAIL] {fname}: {e}")

        test_results.append(test_entry)

    limitations = [
        "1. Single-Image Tracking Limitation: ByteTrack allocates initial detection IDs on a static image. Temporal tracking continuity, trajectory projection, and MOTA metrics require sequential multi-frame video and cannot be evaluated or claimed from single images.",
        "2. Ground-Truth Data Limitation: The local test images do not have curated ground-truth bounding box XML/JSON or lane annotation masks. Benchmark accuracy metrics (mAP@0.5, MOTA, precision, recall) are strictly omitted to prevent fabrication; empirical detection counts and latencies are reported.",
        "3. UFLD Pretrained Domain: The Ultra-Fast Lane Detector is pretrained on the TuSimple highway benchmark. Urban crossroads, sharp 90-degree turns, or roads without painted line boundaries may exhibit reduced lane confidence or fallback gridding anchors.",
        "4. Traffic Light Heuristic Baseline: The Version 1 classifier utilizes spatial-chrominance and HSV heuristics. While effective on standard vertical signals, severe backlight glare, unlit lamps, or horizontal configurations may produce 'unknown' or ambiguous classifications. (Deep learning fine-tuning is planned for Version 2).",
        "5. Pretrained General Detector: YOLO11n is trained on COCO (general classes). Small, distant, or heavily occluded vehicles may fall below the detection confidence threshold (0.25).",
    ]

    summary_report = {
        "suite_name": "Version 1 Perception Module Static Image Test Suite",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_device": device,
        "total_tests": len(test_cases),
        "successful_tests": successful_count,
        "failed_tests": failed_count,
        "pass_rate_percent": round(100.0 * successful_count / max(1, len(test_cases)), 1),
        "test_results": test_results,
        "limitations": limitations,
    }

    # Save summary report
    summary_path = config.METRICS_DIR / "image_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(" PERCEPTION IMAGE TEST SUITE SUMMARY")
    print("=" * 70)
    print(f" Total Tests Run:   {len(test_cases)}")
    print(f" Successful Tests:  {successful_count}")
    print(f" Failed Tests:      {failed_count}")
    print(f" Pass Rate:         {summary_report['pass_rate_percent']}%")
    print("-" * 100)
    print(f"{'Image Filename':<22} | {'Stat':<4} | {'Modules':<7} | {'ms':>7} | {'Detections Summary'}")
    print("-" * 100)
    for res in test_results:
        dets = res["detections"]
        mods = res.get("modules_requested", "all")
        if res["status"] == "PASS":
            d_str = f"Veh:{dets.get('vehicles', 0)}, Ped:{dets.get('pedestrians', 0)}, TL:{dets.get('traffic_lights', 0)}, Lanes:{dets.get('active_lanes', 0)}"
            print(f"{res['filename']:<22} | {res['status']:<4} | {mods:<7} | {res['latency_ms']:>7.1f} | {d_str}")
        else:
            print(f"{res['filename']:<22} | {res['status']:<4} | {mods:<7} | {'N/A':>7} | Error: {res['error']}")
    print("-" * 100)

    # Print visual verification note
    print("\n" + "-" * 100)
    print(" VISUAL VERIFICATION STATUS")
    print("-" * 100)
    print(" PASS = execution completed without error (code ran, image saved)")
    print(" Visual correctness requires human inspection of the output images.")
    print(" No benchmark metrics (mAP, MOTA, precision, recall) are claimed.")
    print("-" * 100)
    for res in test_results:
        if res["status"] == "PASS":
            vv = res.get("visual_verification_target", "")
            print(f" {res['filename']:<22} -> Target: {vv}")
            print(f" {'':22}    Result: {res.get('visual_verification_result', 'NOT VERIFIED')}")

    print("\nIdentified System Limitations:")
    for lim in limitations:
        print(f"  - {lim}")
    print(f"\nAnnotated result images saved in: {config.RESULT_IMAGES_DIR}")
    print(f"Full test summary JSON saved to: {summary_path}")
    print("=" * 70)

    return summary_report


def main():
    parser = argparse.ArgumentParser(description="Autonomous Driving Perception System - Version 1")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input image for single-image perception inference")
    parser.add_argument("--modules", "-m", type=str, default="all", help="Perception modules to execute: 'all', or comma-separated 'A,B,C' / 'objects,traffic_light,lanes'")
    parser.add_argument("--output-image", "-o", type=str, default=None, help="Custom output path for annotated image (default: outputs/result_images/<name>_annotated.jpg)")
    parser.add_argument("--crop", action="store_true", help="Treat input image directly as a cropped traffic light image")
    parser.add_argument("--test-images", action="store_true", help="Run comprehensive image perception test suite on all curated local test images")
    parser.add_argument("--video", type=str, default=None, help="Path to input driving video (default: Wyamo Car video.mp4)")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum number of video frames to process (None for all)")
    parser.add_argument("--test-all", action="store_true", default=False, help="Run legacy static benchmark tests")
    parser.add_argument("--skip-video", action="store_true", help="Skip video processing")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    # Mode 1: Individual Image Inference
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            # Check test_data directory fallback
            cand = config.TEST_DATA_DIR / args.image
            if cand.exists():
                img_path = cand
            else:
                print(f"[Error] Image not found at '{args.image}' or '{cand}'.")
                sys.exit(1)

        print("\n" + "=" * 65)
        print(" Individual Image Perception Inference")
        print(f" Input Image:      {img_path}")
        print(f" Selected Modules: {args.modules}")
        print(f" Target Device:    {args.device}")
        print("=" * 65)

        pipeline = PerceptionPipeline(device=args.device)
        annotated_img, telemetry = pipeline.process_image(
            image_input=img_path,
            output_image_path=args.output_image,
            modules=args.modules,
            is_crop=args.crop,
        )

        # Print clean empirical results
        print("\n" + "-" * 40)
        print(" Inference Results")
        print("-" * 40)
        print(f" Image Resolution: {telemetry['resolution']}")
        print(f" Processing Time:  {telemetry['latency_ms']:.1f} ms ({telemetry['fps_equivalent']:.1f} FPS equiv.)")
        print("\n Modules Executed:")
        for mod, executed in telemetry["modules_executed"].items():
            status_tag = "EXECUTED" if executed else "SKIPPED"
            print(f"   - {mod:28s}: {status_tag}")

        counts = telemetry["detection_counts"]
        print("\n Empirical Detection Counts:")
        print(f"   - Vehicles:        {counts['vehicles']}")
        print(f"   - Pedestrians:     {counts['pedestrians']}")
        print(f"   - Total Objects:   {counts['total_objects']}")
        print(f"   - Traffic Lights:  {counts['traffic_lights']}")
        print(f"   - Active Lanes:    {counts['active_lanes']}")

        # Module specific details
        if counts['total_objects'] > 0:
            print("\n Detected Objects (Module A):")
            for obj in telemetry["module_details"]["module_A"]["tracked_objects"]:
                tid_str = f"#{obj['track_id']} " if obj['track_id'] is not None else ""
                print(f"   * {tid_str}{obj['label']} (conf: {obj['confidence']:.2f}, box: {obj['box']})")
            print("   [Note] Track IDs reflect single-frame tracker initialization.")
            print("          Temporal tracking continuity/MOTA cannot be claimed on single static images.")

        if counts['traffic_lights'] > 0:
            print("\n Detected Traffic Lights (Module B):")
            for tl in telemetry["module_details"]["module_B"]["traffic_lights"]:
                print(f"   * State: {tl['classified_state'].upper()} (conf: {tl['state_confidence']:.2f}, box: {tl['box']})")

        if counts['active_lanes'] > 0:
            print("\n Detected Lanes (Module C):")
            print(f"   * Anchor point counts per lane: {telemetry['module_details']['module_C']['lane_point_counts']}")

        print(f"\n Output Annotated Image Saved To:\n   {telemetry['output_path']}")
        print("=" * 65)
        return

    # Mode 2: Test Suite on All Local Test Images
    if args.test_images:
        run_all_image_tests(device=args.device)
        return

    # Mode 3: Video Pipeline & Legacy Unit Tests (Preserved 100%)
    audit_report = audit_dataset_availability()
    print("\n--- Workspace Dataset & Resource Audit ---")
    print(json.dumps(audit_report, indent=2))

    test_results = {}
    if args.test_all or (not args.image and not args.test_images and not args.video):
        test_results = run_unit_tests_and_static_benchmarks(device=args.device)

    video_summary = {}
    if not args.skip_video:
        video_input = args.video or config.DEFAULT_DRIVING_VIDEO
        if not os.path.exists(video_input):
            print(f"[Warning] Primary video '{video_input}' not found. Searching alternatives...")
            if os.path.exists(config.FALLBACK_DRIVING_VIDEO):
                video_input = config.FALLBACK_DRIVING_VIDEO
            else:
                cand = r"C:\Users\Sriram Siddhartha\Downloads\gemini_generated_video_06bdb8d4.mp4"
                if os.path.exists(cand):
                    video_input = cand

        out_video_name = f"annotated_perception_{Path(video_input).stem}.mp4"
        out_video_path = str(config.ANNOTATED_VIDEOS_DIR / out_video_name)

        pipeline = PerceptionPipeline(device=args.device)
        video_summary = pipeline.process_video(
            video_path=video_input,
            output_video_path=out_video_path,
            max_frames=args.max_frames,
            save_keyframes=True,
            keyframe_dir=str(config.RESULT_IMAGES_DIR),
        )

    final_report = {
        "system_version": "Version 1 (Pretrained Baseline)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_device": args.device,
        "dataset_audit": audit_report,
        "unit_test_results": test_results,
        "video_pipeline_summary": video_summary,
    }

    report_path = config.METRICS_DIR / "perception_v1_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[Execution Complete] Full perception report saved to: {report_path}")


if __name__ == "__main__":
    main()
