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

---

# v1.2 — Module A labelling + object-detection evaluation

## Added

| File | Purpose |
|---|---|
| `scripts/annotate.py` | proposal → render → finalize labelling workflow; proposals stored separately from verified GT; `export-yolo` for use with LabelImg/CVAT |
| `scripts/eval_detection.py` | per-class AP@0.5, P/R at the deployment threshold, 1000-resample paired bootstrap over images, timestamped run directory |
| `scripts/eval_tracking.py` | tracking-continuity proxy (class-filtered vs pre-v1.1 all-class tracking); explicitly not MOTA |
| `labels/detection/manifest.json` | image list with scene ids and dev/eval split |
| `labels/detection/proposals/*.json` | 14 files of **unverified** model suggestions |
| `labels/detection/gt/*.json` | 8 files of verified labels (67 instances) with verifier, method and counts |
| `models/yolo11s.pt` | 18.4 MB official YOLO11s weights, for the comparison only (not the default) |

## Modified

| File | Change | Why |
|---|---|---|
| `modules/object_detection_tracking.py` | `track(..., filter_classes=True)` switch | A/B the class-filtered tracking change with identical code paths |
| `scripts/eval_detection.py` | warm-up inference excluded from timing | the first image absorbed graph setup and produced a bogus p90 of 691 ms |
| `EVALUATION.md` | section 7 added | measured results, prototype-level caveats |
| `README.md` | Module A claims updated; labelling/eval commands added | the previous text said no detection metric was possible |

## Measured outcome
Lanes: neither checkpoint is suitable (TuSimple 58.4 px mean error, CULane 77.0 px and zero lanes
found on the curved road); TuSimple stays the default. Traffic lights: 7/11 held-out correct vs
6/11 for v1.0, but green-to-yellow confusion (3/6) and weak abstention (1/6 unlit) remain. The
cause is housing colour winning over the small lamp; it was not patched because that would tune
on the held-out set. The sample driving clip contains no lane markings at all, so lane behaviour
on it is not scoreable.

## Notes
* `models/culane_18.pth` (178 MB) is untracked by design (see .gitignore); re-download with the
  Google Drive id in EVALUATION.md 8.3 if needed. Everything else runs without it.
* Passing regression tests mean 'behaved as specified', not 'detected correctly'.
---

# v1.4 — traffic-light lamp vs housing, honest lane ground truth

Commit range: `6c79bbe..HEAD`. Baseline `24c9cd0` untouched.

## The bug

`TrafficLightClassifier` computed its hue histogram over the **entire chromatic
mask** while selecting the **largest component** for its geometry stats. In a
detector box the signal body is the largest chromatic region, so a cream/yellow
housing won over a lit green lamp: crop 39 returned "yellow (0.80)" from 16 661
housing pixels. Worse, yellow paint and amber glass have identical chromaticity
(`min(R,G)-B` = 0.60-0.65 for both), so four *unlit* housings came back "yellow"
at 0.81-0.92 confidence.

## The fix

| File | Change | Why |
|---|---|---|
| `modules/traffic_light_classification.py` | rewritten scorer: per-state chroma components, scored by bezel contrast and bezel darkness, with body-size discount, competitor margin and abstention | the old rule asked "what colour is the biggest region"; a lamp is the small region that is *brighter than the bezel around it* |
| same | red index now subtracts the amber excess | held-out run showed amber lamps reported as red (R~G>B satisfies `R-max(G,B)`) |
| same | `TL_MIN_STATE_CONF` collapsed into `TL_MIN_LAMP_SCORE` | two thresholds disagreed, so video abstained where still images did not |
| `config.py` | new thresholds, each annotated with the set it was chosen on; v1.1 constants retained for `--compare` | reproducible comparison against archived classifiers |
| `scripts/eval_traffic_lights.py` | scores dev and held-out separately; loads v1.0/v1.1 from their commits | a tuning set cannot also be a held-out set |
| `scripts/fetch_tl_v2.py` | fetches 12 CC-licensed photographs (6.6 MB) with per-file author and licence | a held-out set must be independent of the images used for tuning |
| `scripts/make_tl_v2_proposals.py` | detector proposals + crops for those photographs | same workflow as before, clearly marked "not labels" |
| `labels/traffic_lights/v2/gt.json` | 18 hand-labelled crops, including an ambiguous all-lamps-lit case | yellow had zero support before this |
| `tests/test_tl_classifier.py` | 14 unit checks | the specific bug, edge cases, glare, unlit housings, voting |
| `scripts/label_lane_rows.py` | paint-trace labeller with overlay output | the previous lane labels were read by eye and were wrong |
| `scripts/eval_lane.py` | boundary error with coverage | coverage matters: the model that "looks best" abstains on the curve |
| `labels/lane/gt.json` | 11 verified boundary points | replaces the withdrawn labels |
| `utils/evaluation.py` | removed the claim that mAP/MOTA were "evaluated via rigorous unit test cases" | that was false |

## Measured outcome

**Traffic lights, dev set (tuning):** 9/11 correct, 0 wrong, 2 abstentions,
6/6 unlit housings abstaining (was 1/6).

**Traffic lights, held-out v2 (n=9 vehicle crops, Wilson 95% CI [0.19, 0.73]):**
the frozen classifier scored 4/9 with 2 amber-as-red errors; the red-index fix
took it to 7/9, and relaxing the amber bezel gate from 0.70 to 0.55 took it to
9/9. Both changes were prompted by evidence outside a clean hold-out (the second
by a synthetic test), so v2 is no longer pristine for amber: the 4/9 is the
held-out number, and 9/9 means "no error in nine crops", not 100% accuracy.

**Lanes:** previous numbers withdrawn (right-boundary labels were 70-110 px off).
On 11 verified points: tusimple/stretch 72.9 px MAE at full coverage;
culane/stretch 49.5 px but no ego pair at all on the curve and zero coverage
under letterbox. Neither is suitable; no replacement model is adopted.

**Integrated:** 4.63 FPS on the 120-frame clip; `sir_image6` green light now
reads GREEN instead of YELLOW. Remaining failures recorded in EVALUATION.md 9.8
and 11: false amber on a US intersection, lane corridors that do not follow the
paint, two missed night signals.

## Notes
* `test_data/tl_v2/*.jpg` is git-ignored (re-fetch with `scripts/fetch_tl_v2.py`);
  the crops, labels and `sources.json` are committed so attribution travels with
  the data. Crops are adaptations of CC BY-SA / CC BY / CC0 works.
* No training was run. The only downloads this round were 6.6 MB of photographs.
* Passing tests still mean "behaved as specified", never "detected correctly".
