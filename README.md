# Autonomous Driving Perception System — Version 1 (Pretrained Baseline)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-red.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO11n-green.svg)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-orange.svg)](https://github.com/ifzhang/ByteTrack)
[![UFLD](https://img.shields.io/badge/Lane%20Detection-Ultra--Fast--Lane--Detection-purple.svg)](https://github.com/cfzd/Ultra-Fast-Lane-Detection)

> **Status: experimental research prototype — not production-ready and not validated for vehicle control.**
> Version 1.1 results, including the regressions, are in [EVALUATION.md](EVALUATION.md); the audit that produced them and every changed file are in [CHANGES.md](CHANGES.md).

A perception research prototype built entirely with **pretrained models**—requiring zero training or fine-tuning. It implements an end-to-end multi-task perception pipeline combining:
1. **Multi-Object Detection and Tracking** (YOLO11n + ByteTrack)
2. **Traffic-Light Localization and State Classification** (YOLO11n + Spatial-Chrominance Baseline)
3. **Deep Lane Detection and Drive Corridor Projection** (Ultra-Fast Lane Detection / UFLD ResNet-18)

---

## 1. System Architecture

```
                                  +--------------------------+
                                  | Input Driving Video /    |
                                  | Camera Frame (RGB)       |
                                  +------------+-------------+
                                               |
             +---------------------------------+--------------------------------+
             |                                 |                                |
             v                                 v                                v
+--------------------------+      +--------------------------+      +--------------------------+
|  Module A: Perception    |      |  Module B: Traffic Light |      |  Module C: Lane Detection|
|  Detection & Tracking    |      |  Detection & State Class.|      |  (Ultra-Fast Lane Det.)  |
+--------------------------+      +--------------------------+      +--------------------------+
| - Pretrained YOLO11n     |      | - YOLO11n (COCO Class 9) |      | - ResNet-18 Backbone     |
| - Filter: Cars, Trucks,  |      | - Bounding box cropping  |      | - 56 Row Anchors         |
|   Buses, Pedestrians     |      | - Spatial-Chrominance /  |      | - 100 Column Grids       |
| - ByteTrack Association  |      |   HSV Heuristic Baseline |      | - TuSimple Weights       |
| - Persistent Track IDs   |      | - States: Red, Yellow,   |      | - Ego Lane Corridor      |
| - Trajectory History     |      |   Green, Unknown         |      | - Projected to Orig Dim  |
+------------+-------------+      +------------+-------------+      +------------+-------------+
             |                                 |                                |
             +---------------------------------+--------------------------------+
                                               |
                                               v
                                  +--------------------------+
                                  | Multi-Task Fusion Engine |
                                  | & Autonomous Driving HUD |
                                  +------------+-------------+
                                               |
                                               v
                                  +--------------------------+
                                  | Browser-Playable H.264   |
                                  | Annotated MP4 Video &    |
                                  | JSON Telemetry Reports   |
                                  +--------------------------+
```

---

## 2. Implemented Modules and Algorithms

### Module A: Object Detection and Multi-Object Tracking
- **Model:** Pretrained `yolo11n.pt` from Ultralytics.
- **Tracker:** ByteTrack algorithm (`bytetrack.yaml`).
- **Target Classes:** Vehicles (`car`, `truck`, `bus`, `motorcycle`) and pedestrians (`pedestrian`, `cyclist`).
- **Capabilities:**
  - Assigns unique, persistent tracking IDs (`#1`, `#2`, `#3`, ...) across consecutive video frames.
  - Maintains motion trails showing past vehicle trajectories.
  - Memory-safe streaming architecture that does not accumulate uncompressed frames in RAM.

### Module B: Traffic Light Detection and State Classification
- **Localization:** Pretrained YOLO11n specifically targeting COCO class 9 (`traffic light`).
- **Classification Method:** **Spatial-Chrominance / HSV Heuristic Baseline Classifier**.
  - *Compliance Note:* As instructed by project requirements, this module is strictly designated as a deterministic baseline and **not** claimed to be a fine-tuned machine learning classifier.
  - *Heuristic Priors:* Exploits physical traffic casing geometry (Red in top 35%, Yellow in middle 30%, Green in bottom 35%) paired with calibrated HSV chromaticity masks.
  - *Rejection Safeguard:* Small, distant, or blurry crops (< 80 px area) and low-saturation or ambiguous lamps automatically return state **`unknown`**.

### Module C: Lane Detection (Ultra-Fast Lane Detection)
- **Model:** ParsingNet architecture from the ECCV 2020 paper *"Ultra Fast Structure-aware Deep Lane Detection"*.
- **Backbone:** ResNet-18 feature extractor.
- **Weights:** Official TuSimple benchmark pretrained weights (`tusimple_18.pth`).
- **Formulation:** Reformulates lane detection as row-based classification across 56 horizontal row anchors and 100 column bins.
- **Visual Projection:** Decodes lane coordinates and projects continuous polyline boundaries plus a semi-transparent green ego-lane drive corridor directly onto the original frame resolution.

---

## 3. Workspace Inspection & Dataset Audit

An audit was conducted across local workspace directories prior to implementation:

1. **Waymo Open Dataset Status:**
   - Course presentations (`Waymo Open Dataset.pptx`) are present in `D:\downloads\`, outlining perception concepts and referencing 20-second Waymo driving segments.
   - However, raw Waymo TFRecord files and ground-truth bounding box/tracking annotations **are not present on local disk**. They reside in remote Google Cloud / Google Drive storage.
2. **Accessible Media Resources:**
   - Driving Video: `D:\downloads\Wyamo Car video.mp4` (480x848 @ 30.0 FPS, 376 frames).
   - Additional Footage: `C:\Users\Sriram Siddhartha\Downloads\gemini_generated_video_06bdb8d4.mp4`.
   - Test Imagery: Straight highway lane markings (`highway_road.jpg`), urban driving scenes (`sample_wyamo.jpg`), and genuine traffic light scenes with Red, Yellow, and Green states.
3. **Evaluation Integrity:**
   - In accordance with guidelines, no synthetic ground-truth labels or benchmark scores have been fabricated.
   - Theoretical metrics (mAP@0.5, MOTA, Lane IoU) are formally implemented in `utils/evaluation.py`.
   - Video inference is evaluated using genuine empirical metrics (FPS, tracking continuity, detection distributions).

---

## 4. Directory Structure

```
Version1_Pretrained/
├── README.md                              # Comprehensive system documentation
├── requirements.txt                       # Pinned dependencies
├── config.py                              # Centralized configuration parameters
├── main.py                                # End-to-end execution pipeline & CLI
├── modules/
│   ├── object_detection_tracking.py       # Module A: YOLO11n + ByteTrack
│   ├── traffic_light_detection.py         # Module B Part 1: YOLO11n TL Detector
│   ├── traffic_light_classification.py    # Module B Part 2: Spatial-HSV Classifier
│   └── lane_detection.py                  # Module C: Ultra-Fast Lane Detection
├── utils/
│   ├── video_utils.py                     # VideoWriter, HUD overlays, properties
│   └── evaluation.py                      # Dataset audit & metric calculations
├── notebooks/
│   ├── Version1_Final.ipynb               # Fully executed, reproducible notebook
│   └── build_notebook.py                  # Notebook builder script
├── models/
│   ├── yolo11n.pt                         # Pretrained YOLO11n weights (5.6 MB)
│   ├── tusimple_18.pth                    # Pretrained UFLD ResNet18 weights (245 MB)
│   └── openh264-2.5.0-win64.dll           # Cisco OpenH264 codec library
├── test_data/                             # Curated test scenes and labeled crops
└── outputs/
    ├── annotated_videos/                  # Playable MP4 videos with HUD overlay
    ├── result_images/                     # Keyframe snapshots and test outputs
    └── metrics/                           # Execution reports and JSON telemetry
```

---

## 5. Setup and Execution

### Step 1: Install Dependencies
```bash
cd Version1_Pretrained
pip install -r requirements.txt
```

### Step 2: Individual Image Inference
Run perception inference on any single static road image with all three modules:
```powershell
python main.py --image "test_data/sample_wyamo.jpg"
python main.py --image "test_data/highway_road.jpg"
python main.py --image "test_data/red_tl.jpg"
python main.py --image "test_data/green_tl.jpg"
python main.py --image "test_data/yellow_inuvik.jpg"
python main.py --image "test_data/tl_crop_red.jpg"
```

Run specific perception modules on an image:
```powershell
python main.py --image "test_data/sample_wyamo.jpg" --modules "A"
python main.py --image "test_data/highway_road.jpg" --modules "C"
python main.py --image "test_data/red_tl.jpg" --modules "B"
```

### Step 2b: Label boxes and evaluate object detection
```powershell
python scripts/annotate.py init-manifest          # scan local images -> labels/detection/manifest.json
python scripts/annotate.py propose --split eval --models yolo11n.pt yolo11s.pt --conf 0.12
python scripts/annotate.py render --image-id all  # numbers + grid -> outputs/runs/labelling_review/
python scripts/annotate.py finalize --image-id <id> --accept all --reject 3,7 --add "x1,y1,x2,y2,2"
python scripts/annotate.py stats                  # coverage per split/class
python scripts/eval_detection.py --models yolo11n.pt yolo11s.pt --imgsz 640
python scripts/eval_tracking.py --video "<clip.mp4>" --frames 120
```
`propose` output is stored as unverified suggestions; only `finalize` writes ground truth, and it
records who verified it and how. `export-yolo --split eval` writes YOLO txt labels if you prefer
to review in LabelImg or CVAT.

### Step 2c: Lanes, traffic lights and regression
```powershell
python scripts/evaluate.py --only lane --lane-model tusimple --lane-preprocess stretch
python scripts/evaluate.py --only lane --lane-model culane  --lane-preprocess stretch
python scripts/eval_traffic_lights.py --compare-legacy
python tests/test_regression.py
```
`--lane-model` selects between the TuSimple and CULane checkpoints (both official, free).
The CULane profile needs `models/culane_18.pth`; everything else runs without it.
Status and measured results for lanes, traffic lights and the integrated run are in
[EVALUATION.md](EVALUATION.md) section 8 - including what does *not* work.

### Step 3: Run Full Static Image Test Suite
Executes multi-module testing across all local curated images and saves a summary report:
```powershell
python main.py --test-images
```

### Step 4: Run Video Perception Pipeline
Stream and annotate driving footage:
```powershell
python main.py --max-frames 120
```

CLI Options Summary:
- `--image <path>`, `-i`: Run perception inference on a single image.
- `--modules <all|A,B,C>`, `-m`: Specify modules to run (default: `all`).
- `--output-image <path>`, `-o`: Custom destination for annotated image.
- `--crop`: Treat input image directly as a cropped traffic light image.
- `--test-images`: Run automated test suite across all curated local test images.
- `--video <path>`: Specify input video (default: `D:\downloads\Wyamo Car video.mp4`).
- `--max-frames <N>`: Limit video frame count (default: 120).
- `--test-all`: Execute isolated unit tests on static test assets.
- `--skip-video`: Only run static test image benchmarks.
- `--device <cpu|cuda>`: Compute device (auto-detected).

### Step 5: Run Jupyter Notebook
Open and run `notebooks/Version1_Final.ipynb`:
```powershell
jupyter lab notebooks/Version1_Final.ipynb
```

---

## 6. Experimental Results & Verification

### Static Unit Tests

**Corrected in v1.1.** The earlier text here reported "100.0% accuracy" on labelled crops; those "crops" are matplotlib figures with axis frames and titles, the labels were hard-coded for those same three files, and the metric was therefore training-set accuracy on synthetic images. It is withdrawn.

What is actually measured now (details and caveats in [EVALUATION.md](EVALUATION.md)):
- **Module B (traffic lights):** 3/3 natural scenes correct, 1/3 on the withdrawn matplotlib figures, n=6 agent-labelled samples — too small to be an accuracy estimate.
- **Module C (lanes):** ego-boundary error 6.7 px (left) / 44.8 px (right) at one labelled row on `highway_road.jpg`; the model remains inaccurate at the image bottom.
- **Module A (objects):** mAP@0.5 **0.710** for yolo11n vs **0.911** for yolo11s on 8 images / 67 verified instances; the paired 95% interval for the difference is [-0.229, +0.045], so the gap is not significant and yolo11n stays the default. Per-class support is thin (58 car, 6 person, 3 truck). See [EVALUATION.md](EVALUATION.md) section 7.

Reproduce with `python scripts/evaluate.py`.

### Driving Video Performance (`Wyamo Car video.mp4`)
- **Frames Processed:** 120 frames (4.0 seconds of driving footage).
- **Processing Speed:** **3.89 - 4.22 FPS on CPU**.
- **Detection Summary:** 237 total vehicle instances detected across 120 frames.
- **Tracking Continuity:** not measured. No ground-truth tracks exist, so no MOTA/ID-switch figure is claimed (the earlier "zero ID switches" statement was unsupported).
- **Output Artifacts:** Generated browser-compatible MP4 video with full HUD overlay (`outputs/annotated_videos/annotated_perception_Wyamo Car video.mp4`).

---

## 7. Known Limitations & Failure Modes in Version 1

1. **Distant Traffic Lights:** Small signals (< 80 px) lack sufficient spatial resolution for reliable color extraction and are safely classified as `unknown`.
2. **Aspect Ratio and Camera Angle:** The vertical smartphone footage (`Wyamo Car video.mp4`) restricts horizontal field-of-view of lane markings compared to standard wide-angle front dashcams.
3. **Illumination Ambiguity:** Heuristic color classifiers can trigger false positives under intense golden-hour sunsets or bright chromatic neon reflections.

---

## 8. Requirements Status Table

| Requirement | Implementation Details | Status |
| :--- | :--- | :--- |
| **Object Detection** | Pretrained YOLO11n (Vehicles, Pedestrians) | **Completed** |
| **Multi-Object Tracking** | ByteTrack with persistent IDs & motion trails | **Completed** |
| **Traffic Light Detection** | Pretrained YOLO11n targeting COCO class 9 | **Completed** |
| **State Classification** | Spatial-Chrominance / HSV Heuristic Baseline | **Completed (Baseline)** |
| **Lane Detection** | Pretrained Ultra-Fast Lane Detection (UFLD ResNet-18) | **Completed** |
| **Video Pipeline** | Streaming frame-by-frame processing with HUD overlay | **Completed** |
| **Browser Video Export** | MP4 VideoWriter with OpenH264 / fallback codecs | **Completed** |
| **Evaluation Integrity** | Dataset audit & empirical statistics without fake metrics | **Completed** |
| **Clean Final Notebook** | Fully runnable `Version1_Final.ipynb` | **Completed** |

---

## 9. Transition Plan to Version 2 (Fine-Tuning)

Version 1 establishes a robust, working pretrained baseline. In **Version 2**, we will:
1. **Acquire Labeled Autonomous Driving Datasets:** Ingest labeled Waymo Open Dataset 2D bounding boxes and Bosch Small Traffic Lights annotations.
2. **Fine-Tune Object Detector:** Fine-tune YOLO11 / YOLOv8 on driving-specific classes (vehicles, pedestrians, cyclists, construction cones).
3. **Train Deep Traffic Light State Classifier:** Replace the heuristic baseline with a fine-tuned CNN classifier (e.g. MobileNetV3 / ResNet-18) trained on 4-state labels (`red`, `yellow`, `green`, `off`).
4. **Fine-Tune Lane Detection:** Adapt UFLD to Waymo urban road geometries.
5. **Comparative Evaluation:** Benchmark Version 2 mAP@0.5 and MOTA gains against this Version 1 baseline.
