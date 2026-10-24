# Changes — Version 1.1 (post-audit)

Baseline preserved as git commit `24c9cd0` ("Baseline: untouched Version1_Pretrained").
Revert any file with `git checkout 24c9cd0 -- <path>`, or everything with `git checkout 24c9cd0 -- .`.
Models were never modified.

## Audit that motivated these changes

| # | Finding | Evidence |
|---|---|---|
| 1 | No ground truth existed; the "100.0 % accuracy" figure came from three matplotlib screenshots used to tune the heuristic | `test_data/tl_crop_*.jpg` are 415x674 figures with axis frames and titles; labels hard-coded in `main.py:585` |
| 2 | `evaluate_detection_map50` / `evaluate_tracking_mota` were dead code | never called anywhere |
| 3 | ByteTrack tracked all 80 COCO classes then filtered | `object_detection_tracking.py:126` |
| 4 | `reset_tracker()` did not reset ByteTrack | measured: IDs resumed at 4..10 after "reset" |
| 5 | Two identical YOLO instances, two forward passes per frame | `main.py:83` and `main.py:90` |
| 6 | Spatial prior assumed the crop is a whole 3-lamp housing; COCO boxes include background; the glare core of a lit lamp has no hue | measured box sizes 369x461, 257x527; night lamp -> `unknown` |
| 7 | Corridor hard-coded to lanes 1/2, polygon built from two independent y-series | `lane_detection.py:238`; self-intersecting fill visible in `outputs/result_images/test_module_C_lanes.jpg` |
| 8 | Every frame stretched to 800x288; 480x848 portrait footage -> near-horizontal lanes | measured 0-1 lanes, corridor inactive on 2/3 video frames |
| 9 | Hard-coded absolute paths | `config.py:79`, `main.py:975`, `evaluation.py:26` |

## Modified files

| File | Change | Why |
|---|---|---|
| `config.py` | added `DETECTION_IMGSZ`, `TL_DETECTION_IMGSZ`, lane geometry/existence/temporal settings, TL blob thresholds; portable `DEFAULT_DRIVING_VIDEO` | thresholds and preprocessing are now explicit, measurable knobs; removes `D:\downloads\...` default |
| `modules/lane_detection.py` | strict checkpoint load; selectable preprocessing (`stretch`/`letterbox`/`crop`); decode uses bin centres and the reference non-existence rule; ego boundaries selected by geometry with a width sanity check; corridor resampled on a shared y-grid; temporal smoothing; occlusion test restricted to vehicle body | fixes the geometric errors and the silent `strict=False` load; all four are measured in `EVALUATION.md` |
| `modules/traffic_light_classification.py` | lamp-blob extraction (largest lit/saturated component), circular hue histogram with per-state tolerance, calibrated reject, weak contradiction veto, temporal voting with a caller-supplied key | whole-crop HSV scoring counted background hue and failed on glare; state is now emitted only after temporal agreement |
| `modules/traffic_light_detection.py` | configurable `imgsz`, duplicate-box merging, `detect_from_boxes()` | recovers small distant signals; removes the second forward pass |
| `modules/object_detection_tracking.py` | `classes=` passed to the tracker; real tracker reset; `imgsz`; records `last_raw_boxes` | fewer association slots wasted, correct per-video reset, shared detection pass |
| `main.py` | shared detection pass for Module B; temporal key per signal; `--imgsz`, `--lane-preprocess`, `--output-video`; device validation with CPU fallback; `resolve_video_path()`; no hard-coded user paths; resets lane/TL state per video | correctness + portability; CLI flags remain backward compatible |
| `utils/evaluation.py` | true per-class AP@0.5 with all-point interpolation; portable dataset audit | the previous "mAP" pooled classes and reported precision/recall at maximum recall |
| `README.md` | claims corrected (see `EVALUATION.md`); setup/eval/demo rewritten | the old README advertised unverifiable results and "production-ready" |
| `.gitignore` | added | keeps 240 MB of weights and generated media out of history |

## Added files

| File | Purpose |
|---|---|
| `scripts/evaluate.py` | reproducible evaluation with environment + config capture, per-module reports |
| `labels/eval_labels.json` | the small agent-labelled set (provenance and caveats in the file) |
| `EVALUATION.md` | before/after numbers, dataset details, limitations, runtime |
| `CHANGES.md` | this file |

## Not done / deliberately out of scope

* No detector comparison (`yolo11n` vs `s`/`m`) was run — it needs labels for a fair comparison, and no box labels exist yet.
* No CNN state classifier was trained — the label set is too small to split by scene.
* No lane-model change (CULane weights) — P2, requires approval.
* Tracking continuity is not measured with MOTA; no GT tracks exist.
