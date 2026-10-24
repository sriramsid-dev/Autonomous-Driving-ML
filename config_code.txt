"""
Configuration settings for Autonomous Driving Perception System - Version 1
Pretrained Baseline Modules:
1. Object Detection & Tracking (YOLO11n + ByteTrack)
2. Traffic-Light Detection & State Classification (YOLO11n + Spatial-Chrominance Baseline)
3. Lane Detection (Ultra-Fast Lane Detection - TuSimple ResNet18)
"""

import os
from pathlib import Path

# Base Paths
PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
TEST_DATA_DIR = PROJECT_DIR / "test_data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

ANNOTATED_VIDEOS_DIR = OUTPUTS_DIR / "annotated_videos"
RESULT_IMAGES_DIR = OUTPUTS_DIR / "result_images"
METRICS_DIR = OUTPUTS_DIR / "metrics"

for d in [ANNOTATED_VIDEOS_DIR, RESULT_IMAGES_DIR, METRICS_DIR, MODELS_DIR, TEST_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hardware / Device
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

# Module A: Object Detection and Tracking
YOLO_MODEL_PATH = str(MODELS_DIR / "yolo11n.pt")
TRACKER_CONFIG = "bytetrack.yaml"
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_IOU_THRESHOLD = 0.45

# COCO Class IDs for Autonomous Driving Context
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light, 11: stop sign
PERCEPTION_TARGET_CLASSES = {
    0: "pedestrian",
    1: "cyclist",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Module B: Traffic Light Detection & State Classification
TRAFFIC_LIGHT_CLASS_ID = 9
TL_DETECTION_CONF_THRESHOLD = 0.20
TL_MIN_CROP_AREA = 80          # Minimum area (pixels) to attempt classification
TL_MIN_WIDTH = 6
TL_MIN_HEIGHT = 10
TL_AMBIGUITY_THRESHOLD = 0.42  # Confidence margin required over competing colors
TRAFFIC_LIGHT_STATES = ["red", "yellow", "green", "unknown"]

# Module C: Ultra-Fast Lane Detection (UFLD)
UFLD_MODEL_PATH = str(MODELS_DIR / "tusimple_18.pth")
UFLD_INPUT_WIDTH = 800
UFLD_INPUT_HEIGHT = 288
UFLD_NUM_GRIDDING = 100
UFLD_NUM_CLS_PER_LANE = 56
UFLD_NUM_LANES = 4
UFLD_ROW_ANCHORS = [
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
    116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
    168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
    220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
    272, 276, 280, 284
]

# Color Palette for Lane Overlays (BGR format)
LANE_COLORS = [
    (255, 105, 65),   # Lane 0: Royal Blue
    (50, 205, 50),    # Lane 1 (Ego Left): Lime Green
    (0, 215, 255),    # Lane 2 (Ego Right): Gold / Yellow
    (220, 20, 60),    # Lane 3: Crimson
]

# Video I/O Defaults
DEFAULT_DRIVING_VIDEO = r"D:\downloads\Wyamo Car video.mp4"
FALLBACK_DRIVING_VIDEO = str(WORKSPACE_DIR / "Wyamo Car video.mp4")
DEFAULT_VIDEO_CODEC = "mp4v"
OPENH264_DLL = str(MODELS_DIR / "openh264-2.5.0-win64.dll")

# Evaluation Settings
IOU_THRESHOLD_EVAL = 0.50
