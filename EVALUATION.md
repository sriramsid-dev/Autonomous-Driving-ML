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
