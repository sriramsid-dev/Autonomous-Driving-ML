"""Calibrate a paint-support test against the hand-verified GT boundaries.

    python scripts/calibrate_paint_support.py

The corridor fix needs to answer one question per row: "is there paint under
this boundary point?" Using a fixed brightness threshold would be a guess. Here
the test is calibrated on the only pixels we know are paint: the 135 points in
labels/lane/clips_gt.json, which were traced on the paint and checked on
overlays.

Variant A (fixed)    v >= 110 and s <= 70, or a yellow range.
Variant B (adaptive) brightness compared with the local road background, so
                     night frames under headlights are not rejected by a
                     daytime constant.

Reported per frame: the fraction of GT rows the mask calls supported. A test
that misses verifiable paint is unusable - it would trim real lanes - so the
bar is high, and any frame it fails on is named rather than averaged away.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GT = ROOT / "labels" / "lane" / "clips_gt.json"
HALF = 18


def mask_fixed(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = ((v >= 110) & (s <= 70)) | ((v >= 90) & (h >= 15) & (h <= 45) & (s >= 60))
    return m.astype(np.uint8)


def mask_adaptive(bgr, win=61, delta=22):
    """Bright relative to the local window median, low saturation."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    vf = v.astype(np.float32)
    local = cv2.blur(vf, (win, win))
    bright = vf >= np.maximum(70.0, local + delta)
    m = (bright & (s <= 110)) | ((h >= 15) & (h <= 45) & (s >= 90) & (vf >= 80))
    return m.astype(np.uint8)


def hit(mask, x, y, half=HALF):
    h, w = mask.shape[:2]
    y, x = int(round(y)), int(round(x))
    if not (0 <= y < h and 0 <= x < w):
        return False
    win = mask[max(0, y - 2):y + 3, max(0, x - half):x + half + 1]
    return bool(win.size) and win.any()


def main() -> int:
    gt = json.loads(GT.read_text(encoding="utf-8"))
    rows = []
    for im in gt["images"]:
        frame = cv2.imread(str(ROOT / im["path"]))
        if frame is None:
            print(f"missing {im['path']}")
            continue
        mf, ma = mask_fixed(frame), mask_adaptive(frame)
        entry = {"path": im["path"], "boundaries": []}
        for b in im["boundaries"]:
            f_hits = [hit(mf, r["x"], r["y"]) for r in b["rows"]]
            a_hits = [hit(ma, r["x"], r["y"]) for r in b["rows"]]
            entry["boundaries"].append({
                "side": b["side"], "kind": b["kind"], "n": len(b["rows"]),
                "fixed": round(float(np.mean(f_hits)), 2),
                "adaptive": round(float(np.mean(a_hits)), 2),
                "fixed_misses": [r["y"] for r, ok in zip(b["rows"], f_hits) if not ok],
                "adaptive_misses": [r["y"] for r, ok in zip(b["rows"], a_hits) if not ok],
            })
        rows.append(entry)
        for b in entry["boundaries"]:
            print(f"{Path(im['path']).parent.name}/{Path(im['path']).stem} {b['side']:5s} "
                  f"{b['kind']:18s} n={b['n']:2d}  fixed {b['fixed']:.0%}  "
                  f"adaptive {b['adaptive']:.0%}"
                  + (f"  fixed misses {b['fixed_misses']}" if b["fixed_misses"] else "")
                  + (f"  adaptive misses {b['adaptive_misses']}" if b["adaptive_misses"] else ""))

    allf = [b["fixed"] for e in rows for b in e["boundaries"]]
    alla = [b["adaptive"] for e in rows for b in e["boundaries"]]
    npts = sum(b["n"] for e in rows for b in e["boundaries"])
    print(f"\n{npts} verified paint points across {len(rows)} frames")
    print(f"  fixed    : mean row support {np.mean(allf):.0%}, "
          f"boundaries fully supported {(np.array(allf) == 1.0).sum()}/{len(allf)}")
    print(f"  adaptive : mean row support {np.mean(alla):.0%}, "
          f"boundaries fully supported {(np.array(alla) == 1.0).sum()}/{len(alla)}")

    out = ROOT / "outputs" / "runs" / "lane_corridor_v17"
    out.mkdir(parents=True, exist_ok=True)
    (out / "paint_support_calibration.json").write_text(
        json.dumps({"half_window_px": HALF, "frames": rows}, indent=1), encoding="utf-8")
    print(f"-> {out / 'paint_support_calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
