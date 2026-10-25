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

## v1.5 — amber paint is not an amber lamp (shape evidence)

Scope: the two `sir_image2` reports the previous round left open. No model was
retrained, no dataset was downloaded beyond 7.3 MB of CC photographs, and every
constant that changed has the measurement that justified it written next to it.

### The defect, and where it actually lived

v1.4 fixed *which region* may be called a lamp (chroma x bezel contrast x bezel
darkness) but not *what shape* a lamp is. On `sir_image2` the three distant
signal heads are all showing green, yet the winning "amber" regions were:

| box | winning amber region | circularity | the lit green lamp in the same box |
|---|---|---|---|
| 1 | 6.4:1 sliver along the housing edge | 0.249 | 0.880 |
| 2 | cream frame round the unlit red lens | 0.140 | 0.733 |
| 4 | paint strip down the housing | 0.116 | 0.882 |

Across the 24 true lit lamps in the dev and v2 crop sets, circularity has a
median of 0.75. A lamp is a disc; housing paint is a crescent, strip or frame.
`_score()` now multiplies by a **compactness** term
(`TL_COMPACT_FLOOR + (1-floor) * min(1, circ / TL_COMPACT_FULL)`, floor 0.25,
full at 0.60). It is weighted, never a hard gate: a genuinely tiny lamp has an
unreliable perimeter, so it is discounted rather than rejected, and any
better-shaped same-state candidate in the same crop still wins on score.

### A second fix that the measurement rejected

With box 2 the remaining competitor is not paint but the **unlit red lens**
(ring contrast 0.15 against the lit lamp's 0.54), because the evidence curve
saturates at contrast 0.15. Raising `TL_RING_FULL_CONTRAST` looked like the
right answer; simulating 0.30 / 0.40 / 0.50 showed it **loses real lamps**
(dev 9 -> 8 -> 7 correct, v2 9 -> 8), since true lit lamps in these sets reach
down to contrast 0.33. Rejected. Box 2 therefore still abstains, which is the
honest outcome for a 23x48 px crop holding one lit lamp beside one unlit lens.

### Measured outcome

**`sir_image2` (integrated):** boxes 1 and 4 now GREEN (were amber at v1.3,
unknown at v1.4); box 2 abstains. No green head is published as amber.

**New untouched set v3** — 11 further CC photographs (`scripts/fetch_tl_v3.py`,
7.3 MB, attribution in `test_data/tl_v3/sources.json`), 38 detector-proposed
regions labelled **by hand** (`scripts/build_tl_v3_gt.py`), materialised after
v1.5 was frozen and scored in a **single** run:

| set | v1.5 closed-set | wrong | abstain | unlit abstain | not-a-light abstain | ambiguous abstain |
|---|---|---|---|---|---|---|
| v3, n=9 signals | 7/9 (0.778) | **0** | 2 | 7/9 | 15/16 | 3/4 |

For comparison on the same untouched crops, v1.1 and v1.0 both score 8/9 with
**1 wrong** and no abstentions at all, and publish a state on 26 of the 29
unlit / not-a-light / ambiguous crops (v1.5: 4 of 29). The older classifiers win
the easy nine-signal count and lose every negative.

**v2 is no longer held out.** It was inspected while fixing the amber bug, so
its current 9/9 is post-hoc and is not quoted anywhere as a held-out estimate.

### Files

| file | why |
|---|---|
| `modules/traffic_light_classification.py` | circularity in `_candidates`, compactness in `_score`, `circularity` in `last_debug` |
| `config.py` | `TL_COMPACT_FLOOR`, `TL_COMPACT_FULL` with the measured justification |
| `tests/test_tl_classifier.py` | 6 new checks -> 20 total: paint stripe, paint frame round a lens, busy background, two heads one lit, lens cut by the edge, real `sir_image2` regression |
| `scripts/eval_traffic_lights.py` | `v3` set registered; note records that v2 is burned |
| `scripts/fetch_tl_v3.py` | reproducible acquisition of the untouched set, no API query needed |
| `scripts/make_tl_v3_proposals.py` | detector proposes regions; labels stay manual |
| `scripts/build_tl_v3_gt.py` | the hand-assigned labels, in reviewable form |
| `labels/traffic_lights/v3/` | crops, `gt.json`, proposals and contact sheet |

### Not claimed
No yellow-accuracy figure beyond "3/3 amber crops in v3, from 2 sources". No
mAP, MOTA or ID-switch number. Two unlit cream heads and one hanging basket
beside a yellow plate are still published as amber.

## v1.6 — lane evaluation on validated clips

The two clips approved in the previous round were both unusable, so this round
found replacements by screening **poster frames only** (no video downloaded),
then labelled and measured.

* `scripts/find_lane_clips.py` — Commons search, poster-frame download, candidate
  metadata. This is how the replacement clips were chosen without spending
  bandwidth on clips that turned out to be drone shots.
* `scripts/fetch_lane_clips.py` — now fetches C (Conor Pass, CC BY-SA 4.0,
  2.5 MB) and D (Ljubljana to Kranj, CC BY 3.0, 62.6 MB); A and B are recorded as
  rejected in the docstring.
* `scripts/extract_lane_frames.py` — the exact 15 frames (10 C/D, 4 extra C, 4 E)
  by frame index or timestamp; clip E's timestamps are used because its webm
  demuxer reports a bogus 1000 fps.
* `scripts/label_lane_marks.py` — seeded paint tracer for white lane paint (row
  median ridge, so dusk-grey paint is not rejected by an absolute S/V test) and
  for yellow centre lines (hue), with an overlay per frame.
* `scripts/build_lane_clip_gt.py` — publishes only the traces that passed the
  overlay check: 8 frames, 9 boundaries, 65 points (`labels/lane/clips_gt.json`).
  Traces onto guardrails, road signs, lamp posts and chevron hatching discarded.
* `scripts/eval_lane.py` — `--gt` so the clip set is evaluated without touching
  the original 11-point photo set.
* `EVALUATION.md` 13-14 — why A/B were rejected, and the four-configuration
  result.

Measured, on the same 8 frames and 65 points for every configuration: tusimple/
stretch coverage 0.477, MAE 94.5 px, median 26.2 px, max 400 px, ego pair 5/8;
tusimple/letterbox 0.385, 224.6 px, ego pair 7/8; culane/stretch and
culane/letterbox **0.000 coverage and 0/8 ego pairs**. No configuration selected:
the one with the most ego pairs has 200 px errors, and the CULane profiles
produce no prediction at all.

## v1.7 — two evaluator defects, two more clips, 13 verified frames

Files touched and why:

* `modules/lane_detection.py` — `detect()` now records `_last_shape`. It was set
  only by `annotate_frame()`, so `scripts/eval_lane.py` split left/right at the
  default 1280x720 centre (x=640) on 1920x1080 frames instead of x=960, and used
  a bottom band starting at y=432 instead of y=648. Measured effect on the same
  8 frames: tusimple/stretch ego pairs 5/8 -> **3/8**, letterbox 7/8 -> **2/8**.
  The §14 table was an artefact of this bug and is superseded in `EVALUATION.md`
  15.1.
* `scripts/eval_lane.py` — `reset()` between frames (the temporal filter was
  blending one scene into the next), plus two new measures: ego-pair frames and
  closest-decoded-lane error. The second separates "never decoded the paint"
  from "decoded it but picked the wrong pair".
* `scripts/fetch_lane_clips.py` — clips F (US 33, CC BY 3.0) and G (I-235 OKC,
  CC BY-SA 4.0) added, both approved before download; 136.8 MB.
* `scripts/find_lane_clips.py` — `--titles` so a named shortlist can be
  poster-screened without downloading the videos.
* `scripts/probe_lane_clip.py` — new: reports frames/fps/duration and writes a
  contact sheet, so a clip is inspected before any labelling.
* `scripts/label_lane_marks.py` — round-2 seeds for F and G. The night frames
  are cropped to y=300..465 because below that the dashboard's glossy
  reflection reads as white paint.
* `scripts/build_lane_clip_gt.py` — round 2 side assignment (US right-hand
  traffic: double yellow is ego's LEFT, white edge its RIGHT) and per-frame
  `unavailable` reasons.
* `labels/lane/clips_gt.json` — expanded from 8 frames / 9 boundaries / 65
  points to **13 frames / 15 boundaries / 135 points**. Refused rather than
  guessed: F02/F03 right (guardrail), G01/G02 left (dashed divider too sparse),
  F04 entirely (centre line in shadow), G03 entirely (gore/merge, ambiguous).
* Diagnostic scripts under `outputs/runs/lane_diag2/` — they load no labels and
  write only to that directory.

Measured on 13 identical frames and 135 points, median / coverage:

| configuration | ego pairs | coverage | median | p90 | max |
|---|---|---|---|---|---|
| tusimple/stretch | 6/13 | 0.363 | 17.0 px | 286 px | 400 px |
| tusimple/letterbox | 6/13 | 0.363 | 156.0 px | 956 px | 1552 px |
| culane/stretch | 3/13 | 0.304 | **7.4 px** | 19 px | 37 px |
| culane/letterbox | 3/13 | 0.104 | 42.7 px | 105.5 px | 142 px |

No configuration selected: CULane is 3-6x more accurate where it fires but covers
3 frames against TuSimple's 6. `min_lane_points` was tested at 3 and 5; lowering
it gained no coverage and doubled the error (13.0 px vs 7.4 px median), so it is
unchanged.

Integrated run in `outputs/runs/integrated_v16/` (8 images + 60 frames of clip F,
both lane models, ~2.8-2.9 FPS on CPU). It exposes a defect the metrics do not
cover: both models draw a phantom lane into the hillside beyond the road and a
corridor that spills onto the field, because the corridor polygon spans the full
overlapping y-range including rows above the labelled band.
