"""Measure ego-corridor validity from the rendered overlay.

    python scripts/eval_corridor.py                       # 13 GT frames + images + video
    python scripts/eval_corridor.py --frames 8 --tag after
    python scripts/eval_corridor.py --json-only

Why the overlay and not the internals: the question is "does the green polygon
describe the lane the vehicle is in", and that has to be answerable the same way
before and after a refactor. Everything here is measured from the annotated
image plus the hand-verified labels, using only the public detector API
(detect / annotate_frame / last_corridor_established), so it runs unchanged
against commit a4b65cb and against the current tree.

Per frame and model, reported:

  corridor     was a polygon drawn at all
  ego          does it cover the pixel straight ahead of the bumper
  top          its highest row, as a fraction of frame height (100% = road only)
  support      fraction of corridor rows with paint under both rendered edges
  spill        share of the corridor area that lies outside the ego lane defined
               by the two GT boundaries (only for frames where both exist)
  drawn        boundaries drawn, and rows dropped as unsupported

Spill is the number that separates a corridor from a green puddle: a corridor
that covers the field scores near 1.0 even when its edges are locally straight.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modules.lane_detection import UltraFastLaneDetector  # noqa: E402

try:  # the pre-fix commit has no paint test at all; measure its output anyway
    from modules.lane_detection import paint_support_mask as _mask_from_module

    MASK_SOURCE = "modules.lane_detection"
    # The helper takes an HSV-like BGR frame; wrap to keep one call signature.
    def paint_mask(frame):
        return _mask_from_module(frame)

    MASK_SOURCE = "modules.lane_detection"
except ImportError:  # pragma: no cover - only for the pre-fix comparison
    MASK_SOURCE = "frozen copy in this script (module has no paint test)"

    def paint_mask(frame, local_win=61, delta=22):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        val_f = val.astype(np.float32)
        local = cv2.blur(val_f, (local_win, local_win))
        bright = val_f >= np.maximum(70.0, local + float(delta))
        return ((bright & (sat <= 110)) |
                ((hue >= 15) & (hue <= 45) & (sat >= 90) & (val_f >= 80)))


def cfg(name, default=None):
    return getattr(config, name, default)


GT_REL = Path("labels") / "lane" / "clips_gt.json"
VIDEO = "F_"

# Original stills only. Scoring against outputs/runs/integrated_*/ frames would
# feed the detector annotated images with boxes and a previous green fill already
# drawn on them - a mistake worth naming, because it looks plausible.
STILLS = [
    ("my_test_images", "sir_image.jpg"),
    ("my_test_images", "sir_image2.jpg"),
    ("my_test_images", "sir_image3.jpg"),
    ("my_test_images", "sir_image7.jpeg"),
    ("my_test_images", "sir_image10.jpeg"),
    ("test_data", "highway_road.jpg"),
    ("test_data", "wyamo_f10.jpg"),
    ("test_data", "wyamo_f200.jpg"),
]
EGO_COL = 0.5      # the camera axis
EDGE_HALF = 12


def corridor_mask(original, annotated):
    """Pixels the corridor fill actually changed, ignoring the 3px lane lines."""
    g0 = original[:, :, 1].astype(np.int16)
    g1 = annotated[:, :, 1].astype(np.int16)
    r1 = annotated[:, :, 2].astype(np.int16)
    b1 = annotated[:, :, 0].astype(np.int16)
    grew = (g1 - g0) > 12
    green = (g1 > r1 + 25) & (g1 > b1 + 25)
    mask = ((grew & green).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask > 0


def edges_by_row(mask):
    """(row, left_x, right_x) for every row the corridor covers."""
    out = []
    for y in np.nonzero(mask.any(axis=1))[0]:
        xs = np.nonzero(mask[y])[0]
        out.append((int(y), int(xs.min()), int(xs.max())))
    return out


def paint_support(frame, rows):
    if not rows:
        return None
    m = paint_mask(frame)
    hits = []
    h, w = m.shape[:2]
    for y, xl, xr in rows:
        ok = False
        for x in (xl, xr):
            win = m[max(0, y - 2):min(h, y + 3), max(0, x - EDGE_HALF):min(w, x + EDGE_HALF + 1)]
            ok = ok or bool(win.size and win.any())
        hits.append(ok)
    return round(float(np.mean(hits)), 3)


def ego_lane_polygon(item):
    """Polygon of the ego lane from the two GT boundaries, or None."""
    sides = {b["side"]: b for b in item["boundaries"]}
    if "left" not in sides or "right" not in sides:
        return None
    lrows = {r["y"]: r["x"] for r in sides["left"]["rows"]}
    rrows = {r["y"]: r["x"] for r in sides["right"]["rows"]}
    ys = sorted(set(lrows) & set(rrows))
    if len(ys) < 2:
        # interpolate the right boundary onto the left rows when both cover them
        ly = sorted(lrows)
        ry = np.array([r["y"] for r in sides["right"]["rows"]])
        rx = np.array([r["x"] for r in sides["right"]["rows"]])
        ys = [y for y in ly if ry.min() <= y <= ry.max()]
        if len(ys) < 2:
            return None
        lx = [lrows[y] for y in ys]
        rx = np.interp(ys, ry, rx)
        return np.array([[x, y] for x, y in zip(lx, ys)] +
                        [[x, y] for x, y in zip(rx[::-1], ys[::-1])], dtype=np.int32)
    return np.array([[lrows[y], y] for y in ys] +
                    [[rrows[y], y] for y in ys[::-1]], dtype=np.int32)


def evaluate(det, frame, name, gt_item=None):
    lanes, ms = det.detect(frame)
    annotated = det.annotate_frame(frame, lanes)
    h, w = frame.shape[:2]
    cmask = corridor_mask(frame, annotated)
    rows = edges_by_row(cmask)
    diag = dict(det.last_corridor_diagnostics or {})
    rec = {
        "name": name,
        "model": det.model_key,
        "shape": [h, w],
        "ms": round(ms, 1),
        "corridor": bool(det.last_corridor_established),
        "reason": diag.get("reason"),
        "drawn_boundaries": sum(1 for (_, why) in (diag.get("lane_draw") or {}).items() if why.startswith("drawn")),
        "decoded_boundaries": sum(1 for l in lanes if len(l) >= 2),
        "unsupported_frac": diag.get("unsupported_frac"),
    }
    if rows:
        top = rows[0][0]
        bottom = rows[-1][0]
        rec["top_frac"] = round(top / h, 3)
        rec["bottom_frac"] = round(bottom / h, 3)
        rec["area_frac"] = round(float(cmask.sum()) / (h * w), 4)
        rec["support"] = paint_support(frame, rows)
        # Camera-axis check at the corridor's OWN bottom row, not at a fixed
        # fraction of the frame: on a left-hand curve the lane is left of the
        # axis higher up, and testing there would call a correct corridor wrong.
        ego_x = int(round(w * EGO_COL))
        y_last, xl_last, xr_last = rows[-1]
        tol = cfg("LANE_EGO_AXIS_TOL_FRAC", 0.0) * max(1, xr_last - xl_last)
        rec["ego"] = bool(xl_last - tol <= ego_x <= xr_last + tol)
        rec["axis_offset_px"] = round((ego_x - 0.5 * (xl_last + xr_last)), 1)
        poly = ego_lane_polygon(gt_item) if gt_item else None
        if poly is not None:
            gt_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(gt_mask, [poly], 1)
            # Compare only over rows the labelled lane actually covers. Otherwise
            # a corridor that correctly extends below the lowest labelled row is
            # charged as "spill", which is a property of the labels, not the
            # corridor.
            y0, y1 = int(poly[:, 1].min()), int(poly[:, 1].max())
            band = np.zeros((h, w), bool)
            band[y0:y1 + 1] = True
            corr_band = int(np.logical_and(cmask, band).sum())
            inside = int(np.logical_and(cmask, gt_mask > 0).sum())
            rec["spill"] = round(1.0 - inside / corr_band, 3) if corr_band else None
            rec["spill_rows"] = [y0, y1]
            rec["corr_rows_in_band"] = corr_band
            rec["gt_lane_present"] = True
        else:
            rec["gt_lane_present"] = False
    return rec, annotated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tusimple", "culane"])
    ap.add_argument("--frames", type=int, default=6, help="frames sampled from clip F")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--data-root", default=None,
                    help="where labels/frames live (default: this checkout). Used to "
                         "score an older checkout's code against the current data.")
    args = ap.parse_args()

    data = Path(args.data_root).resolve() if args.data_root else ROOT
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or stamp
    out_dir = data / "outputs" / "runs" / "lane_corridor_v17" / f"eval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"code from {ROOT}\ndata from {data}\npaint mask: {MASK_SOURCE}")

    gt = json.loads((data / GT_REL).read_text(encoding="utf-8"))
    jobs = []
    for item in gt["images"]:
        frame = cv2.imread(str(data / item["path"]))
        if frame is not None:
            jobs.append(("gt", item, frame, Path(item["path"]).stem))
    if not args.no_images:
        for rel, name in STILLS:
            p = data / rel / name
            f = cv2.imread(str(p)) if p.exists() else None
            if f is not None:
                jobs.append(("img", None, f, Path(name).stem))
    vids = sorted((data / "test_data" / "lane_clips").glob(f"{VIDEO}*.webm"))
    if vids and args.frames:
        cap = cv2.VideoCapture(str(vids[0]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(5, max(6, total - 5), args.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                jobs.append(("vid", None, f, f"F_{int(idx):05d}"))
        cap.release()

    print(f"{len(jobs)} frames x {len(args.models)} models")
    all_recs = []
    for key in args.models:
        det = UltraFastLaneDetector(model_key=key)
        if not det.is_loaded:
            print(f"skip {key}: {det.load_error}")
            continue
        print(f"\n=== {key} ===")
        recs = []
        for kind, item, frame, name in jobs:
            det.reset()
            rec, ann = evaluate(det, frame, name, item)
            rec["kind"] = kind
            recs.append(rec)
            cv2.imwrite(str(out_dir / f"{key}_{name}.jpg"), ann)
            c = "YES" if rec["corridor"] else " no"
            extra = ""
            if rec["corridor"]:
                spill = rec.get("spill")
                extra = (f" top {rec.get('top_frac', 0):.0%}  ego {'ok' if rec.get('ego') else 'NO'}"
                         f"  support {rec.get('support')}"
                         + (f"  spill {spill:.0%}" if spill is not None else ""))
            else:
                extra = f"  {rec['reason']}"
            print(f"  [{c}] {name:26s} drawn {rec['drawn_boundaries']}/{rec['decoded_boundaries']}"
                  f"  {extra}")
        all_recs.append({"model": key, "frames": recs})

    summary = {}
    for block in all_recs:
        recs = block["frames"]
        est = [r for r in recs if r["corridor"]]
        spilled = [r for r in est if (r.get("spill") or 0.0) > 0.25]
        summary[block["model"]] = {
            "frames": len(recs),
            "corridors": len(est),
            "corridor_rate": round(len(est) / max(1, len(recs)), 3),
            "ego_covered": sum(1 for r in est if r.get("ego")),
            "with_gt_pair": sum(1 for r in recs if r.get("gt_lane_present")),
            "spill_gt_available": sum(1 for r in est if r.get("spill") is not None),
            "major_spills": len(spilled),
            "mean_spill": (round(float(np.mean([r["spill"] for r in est
                                                if r.get("spill") is not None])), 3)
                           if any(r.get("spill") is not None for r in est) else None),
            "mean_support": (round(float(np.mean([r["support"] for r in est if r.get("support")])), 3)
                             if any(r.get("support") is not None for r in est) else None),
            "mean_top_frac": (round(float(np.mean([r["top_frac"] for r in est])), 3)
                              if est else None),
            "boundaries_drawn": sum(r["drawn_boundaries"] for r in recs),
            "boundaries_decoded": sum(r["decoded_boundaries"] for r in recs),
        }
        s = summary[block["model"]]
        print(f"\n{block['model']}: {s['corridors']}/{s['frames']} corridors, "
              f"ego-covered {s['ego_covered']}, spills>25% {s['major_spills']}"
              f"/{s['spill_gt_available']}, mean spill {s['mean_spill']}, "
              f"support {s['mean_support']}, top {s['mean_top_frac']}, "
              f"drawn {s['boundaries_drawn']}/{s['boundaries_decoded']}")

    (out_dir / "corridor_eval.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git_note": "run against the current tree; see EVALUATION.md 16 for the pair",
        "code_root": str(ROOT),
        "data_root": str(data),
        "labels": str(GT_REL),
        "paint_mask_source": MASK_SOURCE,
        "config": {
            "paint_support": cfg("LANE_PAINT_SUPPORT"),
            "ego_anchor": cfg("LANE_REQUIRE_EGO_ANCHOR"),
            "max_bottom_gap_frac": cfg("LANE_CORRIDOR_MAX_BOTTOM_GAP_FRAC"),
            "min_rows_frac": cfg("LANE_CORRIDOR_MIN_ROWS_FRAC"),
        },
        "summary": summary,
        "results": all_recs,
    }, indent=1), encoding="utf-8")
    print(f"\n-> {out_dir / 'corridor_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
