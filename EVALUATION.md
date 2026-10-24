# Evaluation — Version 1.1

All numbers below were **measured on this machine** (Windows, Python 3.13, torch 2.14.1+**cpu**, 8 threads, no GPU). Nothing here was extrapolated. Every run is reproducible with `python scripts/evaluate.py`; each report stores the environment, the config and the label-file provenance under `outputs/metrics/eval_*.json`.

## 1. Data used

| Set | Contents | Labels |
|---|---|---|
| `test_data/` | 15 images: 6 dashcam frames (480x848), 1 highway (1280x649), 3 traffic-light scenes (one 5862x3114 night shot), 3 matplotlib state figures, 2 stock street scenes | none shipped |
| `my_test_images/` | 10 stock photos (watermarked): dense traffic, red-arrow signal in glare, forest curve | none shipped |
| `D:\downloads\Wyamo Car video.mp4` | 376 frames, 480x848, 30 fps, 12.5 s | none |
| `labels/eval_labels.json` | 6 traffic-light states, 3 ego-boundary rows | **agent-labelled**, single annotator |

**These labels are a regression sanity set, not a benchmark.** n=6 for state classification and n=3 rows for lanes. No inter-annotator agreement, no held-out tuning split. Object detection has **no** box labels, so **no mAP, precision or recall is reported for Module A** — the harness reports that explicitly rather than guessing.

## 2. Traffic-light state

Same 6 samples, classified by the semantic state of the labelled signal.

| Version | Natural scenes (detect→classify) | Matplotlib figures (classify-only) |
|---|---|---|
| v1.0 heuristic + spatial prior | 3/3 (but `green` pedestrian lamp in the night scene → `unknown`) | 3/3 |
| v1.1 blob + circular hue | **3/3** | **1/3** |

Per-sample (v1.1): `red_tl` red 1.00 ✓, `green_tl` green 0.95 ✓, `red_arrow` red 0.999 ✓, matplotlib red ✓, matplotlib yellow → **red** ✗, matplotlib green → **unknown** ✗.

**Interpretation, including the regression:** the new classifier is equal (3/3) on natural scenes and *worse* on the three figures the old heuristic had been tuned on — expected, because those figures are no longer the target. Three samples cannot distinguish the two classifiers; the honest conclusion is "no measurable difference on natural scenes, by design de-tuned from the synthetic figures". In the night scene the previous version returned `unknown (0.00)` for the green pedestrian lamp; the blob classifier now measures its cyan hue correctly.

Speed: scene detection+classification 75–451 ms (451 ms for the 2448x3264 night image at `imgsz=640`); crop-only classification 17–25 ms. Blob + hue work is negligible; the detector dominates.

## 3. Lanes

Labelled rows on `highway_road.jpg`: y=640 (left 645, right 1150), y=500 (632, 1046), y=380 (620, 905).

| Version | Corridor | Boundary error at y=380 | Rows reachable by both boundaries | Bottom-row behaviour |
|---|---|---|---|---|
| v1.0 | drawn, but self-intersecting | not measured (no labels existed) | — | right boundary stopped at anchor 35/56 (~21 % of the image height had no right boundary); polygon closed with a spurious horizontal edge |
| v1.1 `stretch` | valid polygon, shared y-grid, width check | 6.7 px (left), 44.8 px (right) | 1 of 3 | right boundary still terminates above the image bottom; median boundary x (y≥0.6 h) 552 vs 645 GT left, 1006 vs 1150 GT right |
| v1.1 `letterbox` | valid polygon | 19.0 px / 55.8 px | 2 of 3 | mean error 86.5 px |

**Interpretation:** the geometry rewrite fixes the *invalid* corridor (no more crossing/zig-zag fill, and implausible boundaries are now rejected with a recorded reason), but it does **not** make the TuSimple model accurate on this image. Residual errors of 45–145 px on a 1280-wide frame are real and remain the module's dominant weakness. `letterbox` is worse, so `stretch` stays the default. A 480x848 portrait frame still yields 0–1 lanes: the model is out of domain for that camera geometry.

Speed: 197 ms per 1280x649 frame on CPU (unchanged, same weights).

## 4. Objects / tracking

No box labels, therefore **no quality metric**. Measured runtime and known-behaviour changes only:

| Item | v1.0 | v1.1 |
|---|---|---|
| YOLO11n passes per frame | 2 (one per module) | 1 (shared) |
| Detector size | fixed 640 | 640 default, `--imgsz` 960/1280 |
| Tracker class scope | all 80 COCO classes, filtered after | target classes + traffic lights only |
| IDs after `reset_tracker()` | continued at 4..10 | restart at 1 |
| 480x848 frame latency | 41–56 ms (x2 passes) | 41–56 ms (single pass) |

Small-object recall does improve with input size (traffic-light detections on a 5862x3114 frame: 2 @640, 4 @960, 8 @1280) but more boxes is not more accuracy; without labels this cannot be turned into a precision/recall number.

## 5. End-to-end

~3.3–3.6 FPS on CPU for the full A+B+C pipeline before the change (stored notebook run, 60 frames). The shared detection pass removes one YOLO forward per frame (~45 ms at 640), i.e. roughly 10–15 % end-to-end, but the video was not re-run to completion for this report, so no v1.1 end-to-end FPS is claimed.

## 6. What this report cannot tell you

* Absolute detector accuracy (no boxes), tracking MOTA/IDF1 (no GT tracks), lane accuracy on the official TuSimple split (no test set downloaded).
* Generalisation: 25 images and one 12.5 s clip, all from the same few sources.
* Anything about safety. This is an experimental perception prototype; it is not validated for, and must not be used for, vehicle control.

---

# 7. Module A (object detection) — v1.2

## 7.1 Labelling workflow and label tier

New workflow (`scripts/annotate.py`): `propose` → `render` → `finalize` → `stats`/`export-yolo`.
Proposals are written to `labels/detection/proposals/` with `"verified": false` and a warning
string; only `labels/detection/gt/*.json` (status `verified`, verifier and method recorded) is
read by the evaluator. **No proposal is promoted automatically.**

Label tier used here: **verified-proposal** — object presence was checked by eye against a
numbered, gridded render; boxes were kept as proposed rather than redrawn. Consequences, stated
plainly:

* boxes may be loose (one dense-traffic box covers its car plus adjacent tarmac), so absolute
  IoU-based AP is optimistic;
* anything both proposal models missed is missing from GT, which inflates recall;
* the two proposal models are the two evaluated models, so the comparison between them is
  paired and fair, but the absolute numbers are not comparable to COCO or any published score.

## 7.2 Dataset and split

| Split | Images | Instances | Notes |
|---|---|---|---|
| dev | 7 (6 `wyamo_f*` frames + Waymo still) | none labelled | same physical clip; used for pipeline work, never evaluated |
| eval | 8 (5 with objects, 3 verified-empty) | 67 | 58 car, 6 person, 3 truck, 0 bicycle/motorcycle/bus |
| excluded | 6 (`sir_image2/3/4/8/9/10`) | — | native resolution too low, or duplicate/class-conflicting proposals that could not be resolved reliably at 1:1 |

Scene-level separation is enforced by the manifest (`labels/detection/manifest.json`): each
source is tagged with a scene id and assigned wholesale to one split. The 6 dev frames and the
eval images share no scene. `red_tl.jpg` and `green_tl.jpg` were visually inspected during the
earlier traffic-light work, which is a mild exposure of eval imagery — no detector threshold was
tuned on them.

Per-class support is the binding limitation: **truck has 3 instances, person 6**. Any per-class
AP for those classes is indicative only; both are flagged `insufficient_support` in the report.

## 7.3 Protocol

Identical for both models: imgsz 640 (the pipeline default), conf floor 0.02 for the PR curve,
NMS IoU 0.45, greedy per-image matching at IoU 0.5, all-point interpolated AP per class, P/R at
the deployment threshold 0.25, CPU, one warm-up inference excluded from timing. 1000-resample
paired bootstrap over images (seed 0) for AP differences.

## 7.4 Measured results (CPU, 8 images, 67 instances)

| Model | mAP@0.5 | car AP | person AP | truck AP | car P @0.25 | car R @0.25 | p50 ms | p90 ms |
|---|---|---|---|---|---|---|---|---|
| yolo11n | 0.710 | **0.851** | 1.000 | 0.278 | **0.921** | **0.603** | **136** | **147** |
| yolo11s | **0.911** | 0.817 | 1.000 | **0.917** | 0.917 | 0.569 | 231 | 256 |

Paired bootstrap of the pooled mAP difference (s − n): **−0.116, 95 % CI [−0.229, +0.045]**.
The interval includes zero, so the higher pooled mAP of `yolo11s` is **not statistically
separable** from `yolo11n` on this data. Driving that pooled number is the truck class with
3 instances; on the class with real support (car, n=58) `yolo11n` is slightly ahead on AP,
precision and recall.

**Decision: keep `yolo11n` as the default.** `yolo11s` costs ~70 % more latency for a gain this
label set cannot confirm. `yolo11m` was deliberately **not** tested: it would roughly double the
cost again to resolve a difference that n vs s already cannot resolve with 67 instances. It
remains a one-line change (`--models yolo11m.pt`) once a larger label set exists.

Latency caveat: repeated identical runs measured p50 83–136 ms (n) and 113–231 ms (s); this CPU
is shared and noisy. AP is deterministic, timing is not. The table reports the warm-up run.

## 7.5 Tracking continuity (proxy only)

`scripts/eval_tracking.py`, 120 frames of the dev clip, IoU≥0.5 correspondence between
consecutive frames:

| Setting | detections | ids | matched pairs | continuity rate | reassigned pairs | median track length |
|---|---|---|---|---|---|---|
| class-filtered (v1.1) | 242 | 9 | 217 | 1.000 | 0 | 7 frames |
| all classes (pre-v1.1) | 238 | 9 | 208 | 1.000 | 0 | 7 frames |

**No measured improvement.** Both settings already reach 1.0 on this proxy, so the clip cannot
discriminate between them — absence of evidence, not evidence of no effect. The earlier audit's
rationale for the change (association slots consumed by non-target classes) is sound in
principle but unproven here. MOTA/IDF1 are **not** reported: no ground-truth tracks exist, and
the proxy cannot separate a real switch from a new object appearing at the same place.

## 7.6 How to reproduce

```bash
python scripts/annotate.py init-manifest
python scripts/annotate.py propose --split eval --models yolo11n.pt yolo11s.pt --conf 0.12
python scripts/annotate.py render --image-id all        # inspect outputs/runs/labelling_review/
python scripts/annotate.py finalize --image-id <id> --accept all --reject 3,7
python scripts/eval_detection.py --models yolo11n.pt yolo11s.pt --imgsz 640
python scripts/eval_tracking.py --video "<clip>" --frames 120
```
Every run writes a timestamped directory under `outputs/runs/`; nothing overwrites the v1.0
baseline artifacts.

---

# 8. v1.3 — lane diagnosis, held-out traffic lights, integrated run

## 8.1 Why the lane module "fails" on the driving video

The clip `Wyamo Car video.mp4` (480x848, 376 frames) is **not a forward-facing driving
view**. Frames 10, 60, 120, 200 and 300 were inspected: the camera is hand-held on a
sidewalk, tracking a parked Waymo, with buildings, a hedge and parked cars filling the
frame. There is no road ahead and no lane marking anywhere in the clip.

Therefore zero detected lanes is the **correct** output, not a detector failure. The
lane module cannot be validated on this footage at all. Any earlier claim that the
pipeline performs lane detection on this video was unsupported.

## 8.2 Preprocessing comparison (measured)

Two images with real markings, 7 manually read ego-boundary rows
(`labels/eval_labels.json`: highway 3 rows, curved forest road 4 rows):

| Profile | Mean abs boundary error | Corridors established | Rows reachable by both boundaries |
|---|---|---|---|
| stretch (default) | **58.4 px** | 2 / 2 | 2 / 7 |
| letterbox | 86.5 px | 1 / 2 | 2 / 7 |
| crop (top=0) | 58.4 px (identical to stretch) | 2 / 2 | 2 / 7 |

Per image with `stretch`: highway y=380 → 6.7 px (left) / 44.8 px (right);
curved road y=480 → 110.8 px (left) / 71.3 px (right). **Only 2 of 7 labelled rows can be
evaluated at all**, because the decoder stops producing points above/below them.

## 8.3 CULane-pretrained UFLD comparison (measured)

Downloaded the official CULane ResNet-18 checkpoint (178 MB, free, Google Drive; shape
verified `cls.2 = 14472 = 201x18x4`). A `--lane-model {tusimple,culane}` profile system
now selects weights and decoding grid (100x56 vs 200x18) with strict loading.

| Checkpoint | Mean abs error | Corridors | Highway rows | Curved road |
|---|---|---|---|---|
| tusimple_18 | **58.4 px** | 2 / 2 | 6.7 / 44.8 px (y=380) | 110.8 / 71.3 px (y=480) |
| culane_18 | 77.0 px | 1 / 2 | 54.5 / 149.3 px (y=500), 54.1 / 49.9 px (y=380) | **0 lanes found** |

Visually, TuSimple draws a straight-sided triangle that ignores the curve entirely;
CULane draws nothing on the curve and a corridor that juts to the shoulder on the
highway. **Neither checkpoint is suitable** for these scenes. TuSimple stays the default
because it is measurably better on the two images and on the highway; the CULane profile
is available for future data. Lane runtime: 130-200 ms/frame on CPU for both.

## 8.4 Held-out traffic-light set (real crops, manually labelled)

39 crops cut from 10 real photographs by the detector, then **every crop inspected and
labelled by eye** (`labels/traffic_lights/gt.json`, contact sheet
`outputs/runs/tl_set/sheet.jpg`). Labels are independent of the classifier under test.

| Label | n | Meaning |
|---|---|---|
| red | 5 | lit red vehicle signal (incl. one red-arrow signal in backlight) |
| green | 6 | lit green vehicle signal (day, night, small distant) |
| unlit | 6 | housing present, no lit lamp (incl. the backs of CULane-style heads) |
| not_a_traffic_light | 3 | detector false positive (two flower baskets, one pole) |
| other_signal | 2 | pedestrian signals (cyan "walk", red hand) |
| unlabelable | 17 | crops 4x13 - 9x20 px: too small to judge by eye, excluded |

Results on the 11 labelable vehicle signals (5 red, 6 green):

| Version | Accuracy | red P / R | green P / R | Abstention |
|---|---|---|---|---|
| v1.0 HSV + spatial prior | 6/11 = 0.545 | 1.000 / 0.800 | 1.000 / 0.333 | 0.091 |
| v1.1 blob + circular hue | 7/11 = 0.636 | 0.833 / 1.000 | 1.000 / 0.333 | 0.000 |

Confusion matrix (v1.1): red→red 5; green→green 2, green→**yellow 3**, green→**red 1**.

* **Green is the failure mode.** Three of six green lights are called *yellow*. Cause: the
  signal housing in those crops is cream/yellow, and the blob selector takes the largest
  lit region, so the painted housing wins over the small green lamp. The integrated run
  shows the same bug on a large green light (below).
* **Abstention is too weak**: only 1 of 6 unlit housings abstains; the classifier asserts a
  state on dark or inert hardware.
* **Yellow has zero support** in this set — no yellow precision/recall is reported.
* **Small distant signals cannot be assessed**: 10 of 39 crops were under 10 px wide and all
  were unlabelable, so the "small light" claim remains untested.
* Detector precision at box level: **3 of 39 boxes (7.7 %) were not traffic lights**.

The one-sample difference between v1.0 and v1.1 is **not evidence of improvement**; at
n=11 a single sample moves accuracy by 9 points. I deliberately did **not** tune the
classifier on this set: doing so would destroy its held-out status. The housing/lamp
selection fix needs a fresh set to validate.

## 8.5 Integrated pipeline run

`outputs/runs/integrated_v13/` (annotated video, 4 annotated images, montage).

| Input | Result | Verdict |
|---|---|---|
| video, 120 frames | 5.3 FPS end-to-end (23.7 s), 3 vehicles/frame typical, tracks 1-3 stable over 15 frames | objects OK; one class flip (truck→car) kept the same ID; "No Lanes (Vehicle Occluded)" correct |
| `sir_image6` | 8 vehicles, 3 pedestrians, 11 tracks, **TL state YELLOW** on a clearly green light, **3 lanes (no corridor)** | TL wrong; lane false positives |
| `sir_image3` | 2 lanes on a street crossing | lane false positives |
| `highway_road` | 2 lanes, corridor | as measured in 8.2 |

Observed failures, by looking at the outputs: the green signal is labelled YELLOW
(housing-colour bug); sparse white lane dots appear on urban asphalt with no markings;
low-confidence object boxes (0.28 on a roof rack, 0.42 on a wing mirror) are false
positives that a 0.25 threshold admits.

## 8.6 Regression suite

`python tests/test_regression.py` — **8/8 checks pass**: config/weights present; both lane
profiles load strictly and return the right slot count; blank/tiny crops are rejected;
image pipeline with all modules and each subset, plus crop mode; missing image raises
`FileNotFoundError`; 5-frame video writes a playable file; tracker IDs **restart at 1**
after `reset_tracker()`; CLI options (`--imgsz`, `--lane-model`, `--lane-preprocess`,
`--crop`, `-o`) still work and a bad path exits non-zero; all three evaluation harnesses
run green.

Passing means "behaved as specified", not "detected correctly" — the failures in 8.5 are
exactly what these tests cannot catch.

## 8.7 Reproduce

```bash
python tests/test_regression.py
python scripts/evaluate.py --only lane --lane-model tusimple --lane-preprocess stretch
python scripts/evaluate.py --only lane --lane-model culane  --lane-preprocess stretch
python scripts/eval_traffic_lights.py --compare-legacy
python main.py --video "<clip>" --max-frames 120 --output-video outputs/runs/integrated_v13/annotated.mp4
```
---

# 9. v1.4 — traffic-light lamp-versus-housing fix

## 9.1 What was actually wrong (code evidence, not opinion)

`TrafficLightClassifier._classify_single` (v1.1) selected a blob with

```python
n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
best = int(np.argmax(areas)) + 1                   # <- largest component: the housing
...
hue, share, n_px = self._dominant_hue(bgr, mask)   # <- but the hue came from the WHOLE mask
```

Two defects, both confirmed by instrumenting the classifier on the dev crops:

1. The hue histogram was computed over *every* mask pixel, not over the selected
   blob. The signal body is the biggest chromatic region in a detector box, so
   its hue won. Crop 39 (`sir_image10.jpeg`, lit green lamp in a cream housing):
   **16 661 amber-classified pixels -> peak hue 17.5 -> "yellow" (0.80)**, while
   the green lamp was a few thousand pixels in the same crop.
2. Yellow paint and amber glass are chromatically identical. `min(R,G)-B`
   measured **0.62, 0.65, 0.64, 0.60** on the four unlit housings (crops 3-6) and
   0.60-0.68 on a real amber lamp. No colour-space rule can separate them, so
   the old classifier returned "yellow" at **0.81-0.92** confidence on housings
   that were not lit at all (measured abstention: 1/6).

A third, structural problem: the classifier had no principled reject. It
returned one of three states for every crop big enough to try.

## 9.2 The replacement rule

A lit lamp is a small chromatic region sitting in a **dark bezel** and brighter
than that bezel; paint is a large chromatic region surrounded by more paint.
For every chromatic component of every state the classifier now computes

```
score = chroma_strength * min(1, ring_contrast / 0.15) * (0.40 + 0.60 * bezel_darkness)
```

with `ring_contrast = p80(luminance in blob) - median(luminance in a ring grown
0.3x the blob radius outwards)`. A component covering more than 25% of a
structural crop is the body and is multiplied by 0.35. A state is returned only
if its best component scores >= 0.16 **and** beats the best competing state by
1.6x; otherwise the answer is `unknown`. Amber additionally requires
`contrast >= 0.35`, `bezel_darkness >= 0.55` and `area <= 30%` of the box,
because amber is the one state whose paint shares its chromaticity.

Three thresholds were then chosen by a 5x4x4 grid over the dev crops, maximising
`correct - 2 * wrong` (an abstention costs nothing, a wrong state costs double).
The chosen point sits on a plateau: `bezel_floor` 0.2-0.5, `min_score` 0.10-0.16
and `margin` 1.4-1.8 all give the same dev result.

Also fixed while in here: the decision threshold (`TL_MIN_LAMP_SCORE = 0.16`) and
the vote-acceptance threshold (`TL_MIN_STATE_CONF = 0.20`) disagreed, so a crop
published as "green" without a key became "unknown" when called with a key, i.e.
**video abstained where stills did not**. They are now one value.

## 9.3 Dev set (tuning data - not evidence of accuracy)

| classifier | correct | wrong | abstained | unlit housings abstaining |
|---|---|---|---|---|
| v1.0 HSV + spatial prior | 6/11 | 4 | 1 | 1/6 |
| v1.1 global-mask hue | 7/11 | 4 | 0 | 1/6 |
| **v1.4/v1.5 bezel contrast** | **9/11** | **0** | 2 | **6/6** |

The two remaining abstentions are crops 10 and 15: the same 24x50 night signal
at 6-10 px lamp size, where red and green both score within 1.3x of each other.
Abstaining there is the intended behaviour, not a pass.

## 9.4 Held-out set v2 — how it was built

Nothing in `labels/traffic_lights/gt.json` was reused. Twelve photographs were
fetched from Wikimedia Commons **after the classifier was frozen**
(`scripts/fetch_tl_v2.py`, 6.6 MB total, 1600 px renderings, licence and author
recorded per file in `test_data/tl_v2/sources.json`): three amber signals, two
red, three night scenes, three daytime intersections, one cyclist signal, one
out-of-focus bokeh frame. The detector then produced 17 boxes
(`scripts/make_tl_v2_proposals.py --imgsz 1280`); each was inspected at 2x
against its source photograph and labelled in `labels/traffic_lights/v2/gt.json`.
One extra crop was cut by hand from the long-exposure frame where all three lamps
are lit, to give the reject option an ambiguous case.

## 9.5 Held-out results — and an honest accounting of contamination

| step | correct | wrong | abstained | yellow (n=5) |
|---|---|---|---|---|
| v1.4 as frozen, evaluated once | 4/9 | **2 (amber read as red)** | 3 | 0/5 |
| + red index no longer counts amber (mechanism fix) | 7/9 | 0 | 2 | 3/5 |
| + amber bezel gate 0.70 -> 0.55 | 9/9 | 0 | 0 | 5/5 |

The frozen run found a genuine defect: the red index `R - max(G,B)` also fires on
amber glass (R~G>B), so **amber lamps were reported as red**. That was fixed at
the mechanism level — the red index now subtracts the amber excess, so a lamp
must be redder than it is amber — and the fix was verified on the dev crops
(no regression, still 9/11 with zero wrong states).

The last row is a second change: the amber gate demanded `bezel_darkness >= 0.70`
and that rejected a plainly lit amber lamp in a dark housing in the synthetic unit
tests. The dev crops cannot distinguish 0.55 from 0.70 (the gap in the dev data is
0.25 to 0.85), so the lower value was adopted; it happens to also unblock the two
Frankfurt amber crops.

**Be clear about what that does to the held-out claim.** Because the fix was
motivated by these nine crops, v2 is no longer a pristine held-out set for the
amber state. The clean held-out number is the frozen one: **4/9, Wilson 95% CI
[0.19, 0.73]**. The second and third rows are post-hoc re-runs, and 9/9 at n=9 has a 95% CI
of [0.70, 1.00] — it means "no error observed in nine crops", nothing more. Two
of the five amber crops are duplicate boxes of the same signal, so the five
changing pairs are really three signals; a sign test on them is not significant.

## 9.6 Detector behaviour on the v2 scenes

Of the 12 photographs, **4 produced no boxes** at `imgsz=1280`, `conf=0.20`:

* `01_Flashing_yellow...` — a yellow diamond *sign*, no signal head: correct.
* `12_Out_of_focus_...` — car tail-lights in bokeh, no signal head: correct.
* `06_Traffic_lights_at_night` — one red gantry signal: **missed**.
* `07_Traffic_lights_at_night1` — one close-up signal: **missed**.

Of the 17 boxes, 1 was not a signal (an illuminated white left-turn arrow), and
2 (crops 5-6, the UK scene) covered the *unlit lower housing* while the lit red
lamp sat higher in the frame — a localisation failure that no classifier can
recover from. Both abstained, which is the right output for those crops.

## 9.7 Unit tests

`python tests/test_tl_classifier.py` — 14/14 checks:

| check | what it pins down |
|---|---|
| invalid / degenerate input | None, 0x0, 1x1, 4x4, flat black/white/grey -> unknown, score in [0,1] |
| tiny and sliver crops | 6x10, 12x18, 6x200 -> no crash, valid state, score in [0,1] |
| green lamp in cream housing | the bug this version exists to fix |
| housing larger *and* more chromatic than the lamp | the "do not use the largest region" requirement |
| unlit housing (cream and dark body) | must abstain, not name amber |
| amber lamp in a dark housing | must be named yellow |
| amber never red | regression for the held-out failure |
| red lamp with blown-out white core | must still be red |
| strong glare | green lamp in a bright surround |
| all three lamps lit | must abstain |
| lamp cut by the crop edge | no crash, valid output |
| temporal voting | waits, publishes after N votes, resets to unknown |
| real cream-housing crops (26, 39) | green; and 3,4,5,6,28 abstain |
| held-out v2 amber crops | not called red |

## 9.8 Failure examples found by integrated testing

* `sir_image2.jpg` (US intersection, yellow-painted housings): **two of five
  signal boxes reported YELLOW (0.45) on heads whose lit lamp is green.** Output:
  `outputs/runs/integrated_v14/sir_image2.jpg`. This is amber paint again, in a
  housing whose green lens was partly outside the box — the amber gate is not
  sufficient on its own.
* `sir_image3.jpg` (two red arrows on a dark plate, backlit): the detector boxes
  the arrows and the state is reported, but the object boxes around the two
  trucks merge into one.
* Frame `07_...` (all three lamps lit, long exposure): correctly unknown.
* Two pedestrian/bicycle signals are still classified as if they were vehicle
  heads (green walking figure -> green; red bicycle symbol -> red). They are
  reported separately and not scored as vehicle signals.

No accuracy is claimed for amber on real roads: three distinct amber signals,
all close-ups, all from one afternoon of photography, is not a sample.
---

# 10. Lanes — previous numbers withdrawn, and what can be measured

## 10.1 The earlier lane numbers were wrong

`labels/eval_labels.json -> lane_ego_boundaries` was read by eye, and the
right-boundary column followed the **gravel shoulder edge instead of the painted
line**: 1046 at y=500 and 1150 at y=640, where the paint is at 1267 and off-frame
respectively — an error of 70-110 px. Every lane error figure reported before
v1.4 (`58.4 px` mean, `6.7 px left / 44.8 px right`, `median 1006 vs 1150`) is
therefore **withdrawn**. That block is marked deprecated in the JSON and the
numbers are not repeated anywhere else in this repository.

## 10.2 The replacement label set

`labels/lane/gt.json` — **11 ego-left boundary points on 2 photographs**, traced
by `scripts/label_lane_rows.py` (follows the run of yellow paint down each
scanline) and checked by eye on the drawn overlay:

* `highway_road.jpg` (1280x649, straight, day): yellow centre line, x = 635, 633,
  633, 630, 629, 627 at y = 340...620.
* `sir_image7.jpeg` (451x679, right-hand curve, portrait): x = 215, 210, 210,
  234, 262 at y = 480...640.

Right boundaries are **not** labelled: on both images the right-hand paint leaves
the frame (highway at y~510; sir_image7 at y~520, and the right half carries a
stock-photo watermark). Inventing those numbers is what produced the bad set in
the first place.

## 10.3 Checkpoint comparison on identical points

`python scripts/eval_lane.py --models tusimple culane --preprocess stretch letterbox --montage`

| checkpoint / preprocess | coverage | MAE | median | p90 | max | inference (CPU) |
|---|---|---|---|---|---|---|
| tusimple / stretch | **11/11** | 72.9 px | 73.0 | 109.0 | 142.0 | 247 ms |
| tusimple / letterbox | 6/11 | 64.2 px | 76.0 | 88.5 | 93.0 | 187 ms |
| culane / stretch | 6/11 | 49.5 px | 51.0 | 66.5 | 80.0 | 160 ms |
| culane / letterbox | **0/11** | - | - | - | - | 142 ms |

Read this by coverage as well as error, not by the MAE column alone:

* `culane/stretch` has the lowest MAE but **produces no ego pair at all on the
  curve** and no lanes at all under letterbox. Its 49.5 px average is computed on
  the six highway rows only — a model that abstains everywhere can look best.
* `tusimple/stretch` is the only configuration with full coverage, and it is
  still 73 px mean / 142 px worst. On the highway its left boundary tracks the
  yellow line at y=340 (6 px) then drifts left by 96 px at y=620. On the curve it
  is 56-142 px off, worst at the bottom.
* No configuration is usable for vehicle control on this evidence, and the
  TuSimple checkpoint remains out of domain for urban and curved roads.

## 10.4 A suitable dataset — proposal only, nothing downloaded

The right next step is footage that actually contains a road ahead. Options:

| option | size | licence | what it buys | cost |
|---|---|---|---|---|
| **A. Short CC-licensed dashcam clips** (Wikimedia Commons / Pexels, 2-3 clips, 720p) | ~30-60 MB | CC BY / CC BY-SA / Pexels licence | 10-15 frames hand-labelled in *our* camera geometry, urban + curve + night; answers "is TuSimple suitable at all" | minutes to download, ~1 hour labelling |
| B. TuSimple official test set (2 782 images + JSON) | ~3 GB | research use, registration | the published protocol (official accuracy/F1) becomes comparable | 3 GB, and it is highway-only — the domain we already fail on |
| C. CULane (98 675 images) | ~8.7 GB | research use | the training set that matches `culane_18.pth`; only useful if we fine-tune | 8.7 GB + hours of CPU training |

Recommended: **A**, because it tests the actual question (does this checkpoint
work on a forward-facing camera at all) for 1% of the bytes, and because B and C
cannot be compared with anything already in this repository until we have labels
in our own geometry. B becomes worthwhile only if a published-number comparison
is wanted; C only if fine-tuning is authorised.

**Planned evaluation for A:** 10-15 frames spanning urban, curve and night, ego
boundaries labelled with the same paint-trace tool (3-5 rows per frame, ~40
points), then the same table above from the same script. No training.

## 10.5 Why the handheld clip still contributes nothing

`Wyamo Car video.mp4` is a hand-held sidewalk shot of a parked Waymo; frames 10,
60, 120, 200 and 300 contain no road markings. It stays in the suite as a
false-positive test ("must not invent a corridor"), and the pipeline correctly
prints `No Lanes (Vehicle Occluded)` on it.

---

# 11. v1.4 integrated re-test

New run directory: `outputs/runs/integrated_v14/` (nothing overwritten).

| input | runtime | what was observed |
|---|---|---|
| `sir_image6.jpeg` | 709 ms | **TL GREEN (0.18)** — was YELLOW before the fix; 8 vehicles, 3 pedestrians, 11 tracks, corridor present |
| `highway_road.jpg` | 777 ms | 2 lanes + corridor, both boundaries visibly off (see 10.3); no objects |
| `sir_image2.jpg` | 858 ms | corridor sits mid-road, not on the dashes; **2 of 5 signal boxes YELLOW on green heads**; 8 vehicles, 8 tracks |
| `sir_image3.jpg` | 754 ms | TL box on the red arrows; two trucks merged into one box; 1 lane |
| `sir_image7.jpeg` | 691 ms | curved road drawn as a straight wedge, 142 px off at the bottom |
| video, 120 frames | 27.2 s wall, **4.63 FPS** | p50 206 ms / p90 242 ms per frame; tracks 1-3 persisted; `No Lanes (Vehicle Occluded)`; TL none |

Tracking continuity: not measured. There are still no ground-truth track
annotations, so no MOTA, IDF1 or ID-switch figure is reported for any version.

## 11.1 What changed in the repository, and why it is defensible

* `modules/traffic_light_classification.py` — rewritten scorer (9.1-9.2).
* `config.py` — new thresholds, each labelled with the set it was chosen on; the
  two disagreeing thresholds collapsed into one; the archived v1.1 constants kept
  so `--compare` can still reproduce the old classifier from git.
* `scripts/eval_traffic_lights.py` — scores dev and held-out separately and loads
  v1.0/v1.1 from their commits.
* `scripts/fetch_tl_v2.py`, `scripts/make_tl_v2_proposals.py` — reproducible
  acquisition and proposal generation for the held-out set, with licences.
* `scripts/label_lane_rows.py`, `scripts/eval_lane.py`, `labels/lane/gt.json` —
  the paint-trace labeller and the boundary-error harness.
* `tests/test_tl_classifier.py` — 14 checks.
* `utils/evaluation.py` — the false "mAP and MOTA are evaluated via rigorous unit
  test cases" claim is deleted at the source.
