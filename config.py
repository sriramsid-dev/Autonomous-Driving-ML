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
TRACKER_CONFIG = os.environ.get("PERCEPTION_TRACKER", "bytetrack.yaml")
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_IOU_THRESHOLD = 0.45
# Detection input size passed to the detector. 640 keeps the previous speed;
# 960/1280 measurably improves small-object recall (see EVALUATION.md) at a
# proportional CPU cost.
DETECTION_IMGSZ = int(os.environ.get("PERCEPTION_IMGSZ", "640"))

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

# Lamp classifier v1.4: per-state chromatic components scored by emission evidence.
# Every value below was fixed on the *dev* crops (labels/traffic_lights/gt.json);
# the v2 held-out set was not used to choose any of them.
# red floor excludes amber paint: on amber (R~G>B) r-max(g,b) sits near 0.2,
# while a red lamp exceeds 0.35.
TL_LAMP_CHROMA_FLOOR = {"red": 0.30, "yellow": 0.24, "green": 0.12}
TL_LAMP_MIN_CHROMA = 0.18      # a chromatic component weaker than this is tint, not a lamp
TL_LAMP_MIN_AREA_FRAC = 0.015  # a lamp is a visible share of the box; below this it is a glint
TL_WORK_SHORT_SIDE = 128       # analysis scale; makes ring radii comparable across crops
TL_MAX_UPSCALE = 8.0           # never invent more detail than this for tiny crops
TL_RING_FULL_CONTRAST = 0.15   # lumen step over the bezel that counts as fully lit
TL_RING_MIN_EVIDENCE = 0.10    # below this the region is flat paint / dark lens / background
TL_BODY_AREA_FRAC = 0.25       # in a structural crop a region this large is the body
TL_BODY_PENALTY = 0.35
TL_BEZEL_FLOOR = 0.40          # weight kept by a blob whose surround is not dark
# Shape evidence, added in v1.5 after the sir_image2 false-ambers: the winning
# "amber" regions there were a 6.4:1 sliver along the housing edge, the cream
# frame round the unlit red lens, and a paint strip down the housing -- all
# crescents (circularity 0.12-0.25) while the lit green lamp in the same box
# measured 0.73-0.88. Across the 24 true lit lamps in the dev and v2 crop sets
# circularity has a median of 0.75. Weighted, never a hard gate: a tiny lamp's
# perimeter is unreliable, so it is discounted instead of rejected.
TL_COMPACT_FLOOR = 0.25        # weight of a completely non-disc-shaped region
TL_COMPACT_FULL = 0.60         # circularity at which full weight is reached
TL_COMPETITOR_MIN_AREA_FRAC = 0.015
TL_YELLOW_MIN_CONTRAST = 0.35  # amber must clear the lamp-on-bezel bar...
TL_YELLOW_MIN_BEZEL = 0.55     # ...inside a genuinely dark surround...
TL_YELLOW_MAX_AREA_FRAC = 0.30 # ...and stay small: amber paint is what fills a box
TL_TIGHT_CROP_SIDE = 40        # <= this short side cannot hold a 3-lamp housing
TL_MIN_LAMP_SCORE = 0.16       # absolute evidence required before naming a state
TL_SCORE_MARGIN = 1.60         # best state must beat the runner-up by this factor
# One threshold, used both to commit a state and to accept its vote. Two
# separate values silently disagreed in v1.4: a crop could be published as
# "green" without a key and rejected as a vote with one, i.e. video abstained
# where single images did not.
TL_MIN_STATE_CONF = TL_MIN_LAMP_SCORE
TL_TEMPORAL_WINDOW = 5         # Frames of state history used for voting
TL_TEMPORAL_MIN_VOTES = 3      # Votes required before a state is published
# Archived v1.1 constants. Only the git-loaded v1.1 classifier used by
# scripts/eval_traffic_lights.py --compare reads these; delete them and the
# comparison run stops working.
TL_BLOB_MIN_SAT = 90
TL_BLOB_MIN_VALUE = 70
TL_LAMP_MIN_PIXELS = 12
TL_DUPLICATE_IOU = 0.55        # Overlapping lamp boxes describing one signal

# Shared detector input size for the traffic-light pass when it runs alone.
TL_DETECTION_IMGSZ = int(os.environ.get("PERCEPTION_TL_IMGSZ", str(DETECTION_IMGSZ)))

# Module C: Ultra-Fast Lane Detection (UFLD)
# Two official checkpoints share the same ResNet-18 parsing head but differ in the
# decoding grid, so the profile selects both weights and decoder geometry.
UFLD_MODEL_PATH = str(MODELS_DIR / "tusimple_18.pth")
LANE_PROFILES = {
    "tusimple": {
        "weights": str(MODELS_DIR / "tusimple_18.pth"),
        "griding": 100,
        "rows": 56,
        "lanes": 4,
        "source": "TuSimple (highway, daytime, marked lanes)",
        "anchors": [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
                    128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184,
                    188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244,
                    248, 252, 256, 260, 264, 268, 272, 276, 280, 284],
    },
    "culane": {
        "weights": str(MODELS_DIR / "culane_18.pth"),
        "griding": 200,
        "rows": 18,
        "lanes": 4,
        "source": "CULane (urban/curved, 9 categories incl. night, crowd, no-line)",
        "anchors": [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287],
    },
}
LANE_MODEL = os.environ.get("PERCEPTION_LANE_MODEL", "tusimple")
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

# V1-FIX: lane geometry / decoding controls
LANE_PREPROCESS = os.environ.get("PERCEPTION_LANE_PREPROCESS", "stretch")  # stretch|letterbox|crop
LANE_CROP_TOP = 0.0            # only used by the 'crop' profile
LANE_EXISTENCE_THRESHOLD = 0.50
LANE_TEMPORAL_SMOOTHING = os.environ.get("PERCEPTION_LANE_TEMPORAL", "1") == "1"
LANE_MIN_WIDTH_FRAC = 0.08     # plausible ego-lane width at the bottom row
LANE_MAX_WIDTH_FRAC = 0.75

# Color Palette for Lane Overlays (BGR format)
LANE_COLORS = [
    (255, 105, 65),   # Lane 0: Royal Blue
    (50, 205, 50),    # Lane 1 (Ego Left): Lime Green
    (0, 215, 255),    # Lane 2 (Ego Right): Gold / Yellow
    (220, 20, 60),    # Lane 3: Crimson
]

# Video I/O Defaults (portable: override with --video or PERCEPTION_VIDEO)
DEFAULT_DRIVING_VIDEO = os.environ.get("PERCEPTION_VIDEO", str(TEST_DATA_DIR / ".." / "demo_video.mp4"))
FALLBACK_DRIVING_VIDEO = str(WORKSPACE_DIR / "Wyamo Car video.mp4")
DEFAULT_VIDEO_CODEC = "mp4v"
OPENH264_DLL = str(MODELS_DIR / "openh264-2.5.0-win64.dll")

# Evaluation Settings
IOU_THRESHOLD_EVAL = 0.50
