"""
Trace ego-lane paint on the extracted clip frames and draw verification overlays.

    python scripts/label_lane_marks.py            # writes draft points + overlays
    python scripts/label_lane_marks.py --show     # only rebuild the overlays

Method, per frame and per boundary:

  1. a seed row near the bottom of the visible road is scanned for paint runs of
     the requested colour (yellow centre line, or white lane paint);
  2. the run nearest the seed is taken, then the trace walks up the frame one
     row band at a time, always choosing the paint run closest to the previous x;
  3. a quadratic is fitted to the collected points and the walk is repeated,
     keeping only rows whose paint run is within a tolerance of the fit;
  4. every surviving point is drawn back onto the frame together with the fit,
     so the result can be checked by eye.

The lanes are not loaded and never touch this script: the trace is computed from
the paint alone, and a boundary is only written into the ground truth after its
overlay has been inspected. Rows where the paint is not visible (a dash gap, the
bonnet, an occluding vehicle) simply produce no point.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "test_data" / "lane_clips" / "frames"
OUT = ROOT / "outputs" / "runs" / "lane_v3" / "overlays"

# Seeds: (frame, side, colour, seed_row, seed_x_hint)
#   seed_row is a row where the marking is known to be present and unoccluded.
#   C clips: Irish road, ego keeps left, so the yellow centre line is ego's RIGHT
#            boundary; the left edge is an unpainted verge and is not labelled.
#   D clips: motorway, dashed white divider on the left and solid white edge on
#            the right.
# (frame, side, colour, seed_row, x_window, y_min, y_max)
#   x_window is where the boundary may legitimately be on any row, which keeps
#   the trace from wandering onto a guardrail or a road sign.
SEEDS = [
    ("C01", "right", "yellow", 740, (380, 950), 470, 815, 639),
    ("C02", "right", "yellow", 760, (380, 950), 470, 815, 622),
    ("C03", "right", "yellow", 795, (380, 950), 470, 815, 610),
    ("C04", "right", "yellow", 795, (380, 950), 470, 815, 632),
    ("C05", "right", "yellow", 795, (380, 950), 470, 815, 635),
    ("D02", "left", "white", 1040, (150, 900), 520, 1045, 465),
    ("D02", "right", "white", 1040, (1100, 1900), 480, 1045, 1700),
    ("D03", "left", "white", 1040, (150, 900), 700, 1045, 450),
    ("D05", "left", "white", 1040, (150, 900), 700, 1045, 260),
    ("D01", "left", "white", 1040, (150, 900), 700, 1045, 483),
    ("D03", "right", "white", 1000, (1100, 1900), 620, 1045, 1626),
    ("D05", "right", "white", 1040, (1100, 1900), 620, 1045, 1574),
    ("D06", "left", "white", 1040, (150, 900), 700, 1045, 366),
    ("D06", "right", "white", 1040, (1100, 1900), 620, 1045, 1643),
    ("C06", "right", "yellow", 795, (380, 950), 470, 815, 620),
    ("C07", "right", "yellow", 795, (380, 950), 470, 815, 625),
    ("C08", "right", "yellow", 795, (380, 950), 470, 815, 628),
    ("C09", "right", "yellow", 795, (380, 950), 470, 815, 620),
    # Round 2 (2026-10-25), clips F and G. US right-hand traffic, so in F the
    # double yellow centre line is ego's LEFT boundary and the white edge line
    # its RIGHT. F is 2560x1440; the road ends by y=1420 (bonnet below).
    # G is 1280x720 at night: the road band is only y=300..465 -- below that the
    # dashboard's glossy reflection would otherwise be traced as white paint.
    ("F01", "left", "yellow", 1350, (500, 1150), 950, 1420, 700),
    ("F01", "right", "white", 1350, (1000, 1450), 1000, 1420, 1290),
    ("F02", "left", "yellow", 1350, (600, 1250), 1000, 1420, 790),
    ("F02", "right", "white", 1350, (950, 1450), 1050, 1420, 1150),
    ("F03", "left", "yellow", 1350, (500, 1250), 1000, 1420, 770),
    ("F03", "right", "white", 1350, (1050, 1450), 1050, 1420, 1330),
    ("G01", "left", "white", 430, (150, 520), 330, 465, 390),
    ("G01", "right", "white", 430, (520, 900), 330, 465, 700),
    ("G02", "left", "white", 430, (150, 520), 330, 465, 390),
    ("G02", "right", "white", 440, (520, 900), 330, 465, 690),
    ("G04", "left", "white", 440, (150, 600), 340, 465, 350),
    ("G04", "right", "white", 440, (560, 950), 340, 465, 720),
]

ROW_STEP = 6           # scan every 6th row
MAX_JUMP = 60          # px, maximum drift from the previous row's run
MIN_RUN = 4            # px, ignore speckle
MAX_RUN = 120          # px, ignore a whole bright vehicle
FIT_TOL = 45           # px, keep points within this of the fitted curve


def paint_mask(bgr, colour):
    """Paint pixels.

    White lane paint is found as a *ridge above the row's own median* rather
    than by an absolute S/V threshold: the dusk motorway frames carry grey-blue
    paint with S well above 70 that an absolute white test rejects, while the
    row median separates paint from asphalt at any exposure. Yellow is found by
    hue, which is stable across daylight and dusk.
    """
    if colour == "yellow":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        m = ((h >= 14) & (h <= 42) & (s >= 80) & (v >= 85)).astype(np.uint8)
    else:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        med = np.median(g, axis=1, keepdims=True)
        m = (g > med + 16).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def runs_on_row(mask_row):
    """(centre_x, width) for every contiguous paint run on one row."""
    idx = np.flatnonzero(mask_row)
    if idx.size == 0:
        return []
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
    out = []
    for r in splits:
        if MIN_RUN <= r.size <= MAX_RUN:
            out.append((float(r.mean()), int(r.size)))
    return out


def _nearest(mask_row, target, window):
    """Paint run centre nearest `target`, restricted to an x window."""
    best, bestd = None, 1e9
    for cx, _wd in runs_on_row(mask_row):
        if not (window[0] <= cx <= window[1]):
            continue
        d = abs(cx - target)
        if d < bestd:
            best, bestd = cx, d
    return best, bestd


def trace(mask, seed_row, x_window, y_min, y_max, seed_x=None, tol=FIT_TOL):
    h, w = mask.shape
    seed_row = min(seed_row, h - 1)
    # snap the seed row to one that actually has paint inside the window
    seed = None
    for dy in range(0, 40, ROW_STEP):
        for y in (seed_row + dy, seed_row - dy):
            if not (y_min <= y <= y_max) or y >= h:
                continue
            hint = seed_x if seed_x is not None else (x_window[0] + x_window[1]) / 2.0
            cx, d = _nearest(mask[y], hint, x_window)
            if cx is not None and d < (65 if seed_x is not None else (x_window[1] - x_window[0]) / 2.0):
                seed, seed_row = cx, y
                break
        if seed is not None:
            break
    if seed is None:
        return []

    ys = list(range(seed_row, max(y_min, 0) - 1, -ROW_STEP))
    ys += list(range(seed_row + ROW_STEP, min(y_max, h - 1), ROW_STEP))
    pts = []
    prev = seed
    for y in ys:
        best, bestd = _nearest(mask[y], prev, x_window)
        if best is None or bestd > MAX_JUMP:
            continue
        pts.append([best, float(y)])
        prev = best
    if len(pts) < 8:
        return pts
    # second pass: refit and keep only points close to the curve
    ys_a = np.array([p[1] for p in pts])
    xs_a = np.array([p[0] for p in pts])
    deg = 1 if len(pts) < 15 else 2
    coef = np.polyfit(ys_a, xs_a, deg)
    keep = [p for p in pts if abs(np.polyval(coef, p[1]) - p[0]) <= tol]
    return keep


def thin(points, step=3):
    """Keep every Nth point, always including the first and last."""
    if len(points) <= 2:
        return points
    keep = points[::step]
    if keep[-1] != points[-1]:
        keep.append(points[-1])
    return keep


def draw(frame, results, out_path):
    vis = frame.copy()
    colours = {"left": (0, 200, 255), "right": (255, 120, 0)}
    for side, pts in results.items():
        col = colours[side]
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 3, col, -1)
        if len(pts) >= 2:
            arr = np.array(pts)
            deg = 1 if len(pts) < 15 else 2
            coef = np.polyfit(arr[:, 1], arr[:, 0], deg)
            ys = np.linspace(arr[:, 1].min(), arr[:, 1].max(), 60)
            for i in range(len(ys) - 1):
                cv2.line(vis, (int(np.polyval(coef, ys[i])), int(ys[i])),
                         (int(np.polyval(coef, ys[i + 1])), int(ys[i + 1])), col, 2)
        cv2.putText(vis, side, (int(pts[0][0]) + 6, int(pts[0][1]) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.imwrite(str(out_path), vis)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "labels" / "lane" / "clips_draft.json"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    frames = {}
    draft = {}
    for fid, side, colour, seed_row, x_window, y_min, y_max, seed_x in SEEDS:
        if fid not in frames:
            img = cv2.imread(str(FRAMES / f"{fid}.png"))
            if img is None:
                print(f"missing frame {fid} - run scripts/extract_lane_frames.py")
                return 1
            frames[fid] = img
        img = frames[fid]
        mask = paint_mask(img, colour)
        pts = thin(trace(mask, seed_row, x_window, y_min, y_max, seed_x), step=3)
        if len(pts) < 6:
            print(f"  {fid} {side:5s}: only {len(pts)} points - not enough to label")
            continue
        entry = draft.setdefault(fid, {"frame": f"test_data/lane_clips/frames/{fid}.png",
                                       "size": [img.shape[1], img.shape[0]],
                                       "boundaries": {}})
        entry["boundaries"][side] = {
            "kind": "yellow_centre_line" if colour == "yellow" else "white_lane_paint",
            "rows": sorted(({"y": int(y), "x": int(round(x))} for x, y in pts),
                           key=lambda r: r["y"]),
        }
        print(f"  {fid} {side:5s}: {len(pts)} points, "
              f"y {int(pts[-1][1])}..{int(pts[0][1])}")

    for fid, entry in draft.items():
        res = {s: [(p["x"], p["y"]) for p in b["rows"]] for s, b in entry["boundaries"].items()}
        draw(frames[fid], res, OUT / f"{fid}_overlay.jpg")
        print(f"overlay -> {OUT / f'{fid}_overlay.jpg'}")

    Path(args.out).write_text(json.dumps(draft, indent=1))
    print(f"\ndraft -> {args.out}  ({len(draft)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
