"""Trace why the corridor polygon goes where it does.

    python scripts/diag_corridor.py                       # all integrated_v16 images
    python scripts/diag_corridor.py --frames 6            # + video frames of clip F

For every image and both checkpoints, prints per lane slot:

  n     decoded points surviving the decode + range + teleport filters
  yspan top..bottom row of the slot, in original pixels
  bx    median x of the points in the bottom 40% of the frame (what the ego
        selector actually uses to decide left vs right)

then the selected ego pair and the polygon it produces: the shared y range, the
corridor top row as a fraction of frame height, and - the part that matters -
how many of the corridor's rows sit on actual paint.

Paint support is measured, not inferred: for every sampled corridor row the
boundary x is looked up and the pixels within +-LANE_SUPPORT_HALF_WIN_PX are
tested with a brightness-plus-low-saturation mask. A row counts as supported
when either boundary has paint under it. The mask is deliberately crude; it is
used to decide *whether the model's evidence is contradicted by the pixels*,
not to detect lanes itself.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modules.lane_detection import UltraFastLaneDetector  # noqa: E402

INTEGRATED = ROOT / "outputs" / "runs" / "integrated_v16"
OUT = ROOT / "outputs" / "runs" / "lane_corridor_v17"
SUPPORT_HALF_WIN = 18      # px either side of the traced boundary
SUPPORT_MIN_ROWS_FRAC = 0.25


def paint_mask(bgr: np.ndarray) -> np.ndarray:
    """Crude 'is this pixel road paint' mask: bright and low saturation."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white = (v >= 110) & (s <= 70)
    yellow = (v >= 90) & (h >= 15) & (h <= 45) & (s >= 60)
    mask = (white | yellow).astype(np.uint8)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def row_supported(mask, x, y, half=SUPPORT_HALF_WIN) -> bool:
    h, w = mask.shape[:2]
    y, x = int(round(float(y))), int(round(float(x)))
    y0, y1 = max(0, y - 2), min(h, y + 3)
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    if y1 <= y0 or x1 <= x0:
        return False
    return float(mask[y0:y1, x0:x1].mean()) > 0.0


def report(det, frame, name, out_dir: Path):
    lanes, ms = det.detect(frame)
    h, w = frame.shape[:2]
    annotated = det.annotate_frame(frame, lanes)
    diag = det.last_corridor_diagnostics

    slots = []
    for idx, pts in enumerate(lanes):
        if len(pts) < 2:
            slots.append({"slot": idx, "n": len(pts)})
            continue
        low = [x for x, y in pts if y >= 0.6 * h]
        slots.append({
            "slot": idx,
            "n": len(pts),
            "yspan": [int(min(y for _, y in pts)), int(max(y for _, y in pts))],
            "bx": round(float(np.median(low))) if low else None,
        })

    row = {"name": name, "shape": [h, w], "ms": round(ms, 1),
           "slots": slots, "diag": diag,
           "corridor": det.last_corridor_established}

    if det.last_corridor_established:
        left, right = det._select_ego_boundaries(lanes)[:2]
        grid = det._common_grid(left, right)
        if grid is not None:
            ys = grid[0][:, 1]
            lx, rx = grid[0][:, 0], grid[1][:, 0]
            mask = paint_mask(frame)
            sup_l = [row_supported(mask, x, y) for x, y in zip(lx, ys)]
            sup_r = [row_supported(mask, x, y) for x, y in zip(rx, ys)]
            either = [a or b for a, b in zip(sup_l, sup_r)]
            row["corridor"] = {
                "y_lo": int(ys.min()), "y_hi": int(ys.max()),
                "top_frac": round(float(ys.min()) / h, 3),
                "height_frac": round(float(ys.max() - ys.min()) / h, 3),
                "area_frac": round(float(np.sum(rx - lx) / (h * w)), 4),
                "support_left": round(float(np.mean(sup_l)), 3),
                "support_right": round(float(np.mean(sup_r)), 3),
                "support_either": round(float(np.mean(either)), 3),
            }

    tag = "OK " if row["corridor"] and isinstance(row["corridor"], dict) else "NO "
    if isinstance(row["corridor"], dict):
        c = row["corridor"]
        print(f"[{tag}] {name:28s} {ms:5.1f}ms  corridor y {c['y_lo']}..{c['y_hi']} "
              f"(top {c['top_frac']:.0%} of h)  support {c['support_either']:.0%}  "
              f"area {c['area_frac']:.1%}")
    else:
        print(f"[{tag}] {name:28s} {ms:5.1f}ms  {diag.get('reason', 'no corridor')}")
    for s in slots:
        if s["n"] >= 2:
            print(f"        slot{s['slot']} n={s['n']:3d} y={s['yspan'][0]}..{s['yspan'][1]} bx={s['bx']}")
    if "left_idx" in diag:
        print(f"        ego left=slot{diag['left_idx']} right=slot{diag['right_idx']} "
              f"width={diag.get('bottom_width_px')}px {diag.get('fallback', '')}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{name}.jpg"), annotated)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="also probe N frames of clip F")
    ap.add_argument("--tag", default="v17_diag")
    args = ap.parse_args()

    out_dir = OUT / args.tag
    results = []
    images = sorted(p for p in INTEGRATED.glob("*.jpg") if "compare" not in p.name)
    for key in ("tusimple", "culane"):
        det = UltraFastLaneDetector(model_key=key)
        if not det.is_loaded:
            print(f"skip {key}: {det.load_error}")
            continue
        print(f"\n=== {key} ===")
        for p in images:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            results.append(report(det, frame, f"{key}_{p.stem}", out_dir))
        if args.frames:
            vids = sorted((ROOT / "test_data" / "lane_clips").glob("F_*.webm"))
            if vids:
                cap = cv2.VideoCapture(str(vids[0]))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i, idx in enumerate(np.linspace(0, total - 1, args.frames).astype(int)):
                    cap.set(cv2.CAP_PROP_FRAME_COUNT, idx)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    det.reset()
                    results.append(report(det, frame, f"{key}_F_f{int(idx):05d}", out_dir))
                cap.release()

    (out_dir / "diag.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    n_ok = sum(1 for r in results if isinstance(r["corridor"], dict))
    print(f"\n{n_ok}/{len(results)} corridors established -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
