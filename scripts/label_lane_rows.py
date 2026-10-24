"""
Trace a painted lane boundary along given rows and draw it for visual verification.

    python scripts/label_lane_rows.py --image test_data/highway_road.jpg \
        --kind white --rows 340 400 460 520 580 620 --x-hint 640 --side right

This is the tool used to produce the lane ground truth in
labels/lane/gt.json. It looks for bright white (or yellow) paint runs per row,
picks the candidate closest to the previous row so the trace follows one line,
and writes an overlay PNG that a human then checks. The trace is *not* produced
by the lane model under test; it never loads the model.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def paint_mask(img, kind):
    b, g, r = [img[:, :, i].astype(np.int16) for i in range(3)]
    v = img.max(axis=2).astype(np.int16)
    if kind == "white":
        return (v > 140) & (np.abs(r - b) < 55) & (np.abs(r - g) < 45) & (np.abs(g - b) < 45)
    return (r > 95) & (g > 80) & (r - b > 45) & (np.abs(r - g) < 85)


def runs(row_mask, min_len=3, max_len=40):
    out, start = [], None
    for i, val in enumerate(row_mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if min_len <= i - start <= max_len:
                out.append((start + i - 1) / 2.0)
            start = None
    if start is not None and min_len <= len(row_mask) - start <= max_len:
        out.append((start + len(row_mask) - 1) / 2.0)
    return out


def trace(img, rows, kind, x_hint, search=90, side="right"):
    mask = paint_mask(img, kind)
    found, prev = {}, x_hint
    for y in sorted(rows):
        cands = runs(mask[y])
        cands = [c for c in cands if abs(c - prev) <= search]
        if side == "right":
            cands = [c for c in cands if c >= prev - search]
        if not cands:
            found[y] = None
            continue
        best = min(cands, key=lambda c: abs(c - prev))
        found[y] = int(round(best))
        prev = best
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--kind", choices=["white", "yellow"], default="white")
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    ap.add_argument("--x-hint", type=int, required=True)
    ap.add_argument("--search", type=int, default=90)
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    img = cv2.imread(str(ROOT / args.image))
    if img is None:
        print(f"cannot read {args.image}")
        return 1
    got = trace(img, args.rows, args.kind, args.x_hint, args.search, args.side)
    for y, x in sorted(got.items()):
        print(f"  y={y:>4}  x={x}")

    vis = img.copy()
    for y, x in got.items():
        if x is None:
            continue
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        cv2.line(vis, (x - 14, y), (x + 14, y), (0, 0, 255), 1)
        cv2.putText(vis, str(x), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out = Path(args.out) if args.out else ROOT / "outputs" / "runs" / "lane_labels" / (
        Path(args.image).stem + f"_{args.kind}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"overlay: {out}")
    print(json.dumps({str(k): v for k, v in sorted(got.items())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
