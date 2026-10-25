# Lane clip inspection — A and B are not usable as lane ground truth

Date: 2026-10-25. Downloaded with `scripts/fetch_lane_clips.py` (120.3 MB, metadata
in `test_data/lane_clips/sources.json`). Nothing else was downloaded.

## What was verified

| clip | licence | size | resolved | frames / duration |
|---|---|---|---|---|
| A `A_dashcam_film.webm` | CC BY-SA 3.0, author Frukko | 80.2 MB | 720x480 @ 29.97 fps | 6278 (209.5 s) |
| B `B_persian_gulf_drive.webm` | CC BY 3.0 | 40.2 MB | 1280x720 @ 25.00 fps | 1473 (58.9 s) |

Both files decode and play. Licences and attribution are recorded in
`test_data/lane_clips/sources.json`; the originals are git-ignored.

## Clip A — forward-facing, but no lane markings anywhere

A is genuinely a dashcam: the camera is on the windscreen, the bonnet and wiper
are visible, there is a burned-in timestamp (05/03/2018). It is a winter drive
through villages and countryside.

I sampled 48 frames evenly across the whole 209 s (`scan_A_48.jpg`) rather than
the 7 probe frames, precisely to avoid judging the clip by one bad minute. Every
one of the 48 shows the same thing: a narrow snow-and-gravel road whose edges
are defined by **snow banks, ploughed ridges and gravel**, with **no painted
lane line, no dashed marking, no centre line** in any frame. Frames 0-40 s are
village streets; 40-209 s are open rural roads with snow banks either side.

Consequence: there is nothing here to label as a lane boundary without
inventing it. Labelling the snow-bank edge would measure "road edge following",
a different task from "lane marking detection", and would silently corrupt any
comparison with the existing TuSimple/CULane paint-based labels.

## Clip B — not a driving camera at all

B is **aerial drone footage** flying along the Makran coast: the camera looks
down at the shoreline, the road appears as a thin ribbon in the middle distance,
and the sea and beach fill most of the frame (`probe_B.jpg`, 7 frames over the
full 59 s). There is no ego lane, no forward-facing vehicle view, and the road
surface is never close enough to read markings.

Consequence: unusable for ego-lane evaluation of any kind.

## What clip A *can* still support

A is the first real *moving* forward-facing driving video in the project (the
existing clip is a handheld shot of a parked car). It is therefore useful for
running-time and robustness checks that do not need lane labels:

* `python main.py --video test_data/lane_clips/A_dashcam_film.webm --max-frames 100`
  processed 100 frames in 29.35 s (**3.48 FPS**) on CPU, tracks persisted, no crash.
* **False lane detections on an unmarked road**: the module reports
  "2 Lanes" and, from frame ~94, "Corridor Active" with a filled corridor drawn
  over the whole carriageway (`A_integrated_frames.jpg`). There is no painted
  line in this scene, so these are false positives. This is a qualitative
  observation, not a metric: with no lane ground truth in the clip, no
  false-positive *rate* can be computed.

## Why I stopped instead of substituting

The instruction was to use A and B only, and not to silently substitute another
dataset. Both are unsuitable for the labelling and four-way comparison that was
requested, so the lane evaluation itself remains **not done**. The existing lane
numbers are unchanged: tusimple/stretch 72.9 px MAE at full coverage over 11
verified points on 2 photographs; culane/stretch 49.5 px but no ego pair on the
curve.

A replacement needs a clip that is (a) filmed from a vehicle, forward-facing,
(b) shows painted lane markings clearly, (c) small and freely licensed. That is
a new download and needs approval.

## Files in this run

| file | what |
|---|---|
| `scan_A_48.jpg` | 48 frames sampled across all of clip A — the evidence that no marking exists |
| `probe_A.jpg`, `probe_B.jpg` | 7-frame probe sheets for each clip |
| `A_integrated.mp4` | 100 annotated frames of clip A through the full pipeline |
| `A_integrated_frames.jpg` | 4 frames from it, showing the false corridor |
