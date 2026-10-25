"""
Lane-boundary error against manually verified painted-line points.

    python scripts/eval_lane.py                          # both checkpoints, stretch
    python scripts/eval_lane.py --models tusimple culane --preprocess stretch letterbox
    python scripts/eval_lane.py --montage                # + overlay images to inspect

Data: labels/lane/gt.json - 11 ego-left boundary points on 2 photographs, read
from the paint itself (see that file's provenance). Only rows where the marking
is actually inside the frame are scored, so "no lane at this row" is reported as
missing coverage rather than counted as a large error.

What this does NOT measure: detection rate over a dataset, night/rain/occlusion,
or anything on the handheld clip in test_data, which contains no road markings
at all and is therefore not lane ground truth.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.lane_detection import UltraFastLaneDetector  # noqa: E402

GT = ROOT / "labels" / "lane" / "gt.json"   # override with --gt


def x_at_row(points, y, tol=12):
    """Interpolate a lane polyline at row y; None when it does not reach that row."""
    if not points:
        return None
    near = [(abs(py - y), px) for px, py in points if abs(py - y) <= tol]
    if near:
        return float(min(near)[1])
    pts = sorted(points, key=lambda p: p[1])
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if y1 <= y <= y2 and y2 != y1:
            t = (y - y1) / float(y2 - y1)
            return float(x1 + t * (x2 - x1))
    return None


def run(model_key, preprocess):
    det = UltraFastLaneDetector(device="cpu", model_key=model_key, preprocess=preprocess)
    if not det.is_loaded:
        return {"model": model_key, "preprocess": preprocess, "error": det.load_error}

    gt = json.loads(GT.read_text(encoding="utf-8"))
    rows_out, errors, misses = [], [], 0
    latencies = []
    for item in gt["images"]:
        frame = cv2.imread(str(ROOT / item["path"]))
        if frame is None:
            continue
        lanes, ms = det.detect(frame)
        latencies.append(ms)
        left, right, _ = det._select_ego_boundaries(lanes)
        per_image = {"path": item["path"], "n_lanes": sum(1 for l in lanes if l), "rows": []}
        for b in item["boundaries"]:
            for r in b["rows"]:
                xs = x_at_row(left if b["side"] == "left" else right, r["y"])
                if xs is None:
                    misses += 1
                    per_image["rows"].append({**r, "model_x": None, "error": None})
                    continue
                err = abs(xs - r["x"])
                errors.append(err)
                per_image["rows"].append({**r, "model_x": round(xs, 1), "error": round(err, 1)})
        rows_out.append(per_image)

    n_pts = sum(len(b["rows"]) for i in gt["images"] for b in i["boundaries"])
    covered = len(errors)
    return {
        "model": model_key,
        "preprocess": preprocess,
        "n_labelled_points": n_pts,
        "covered": covered,
        "coverage": round(covered / n_pts, 3) if n_pts else None,
        "mae_px": round(float(np.mean(errors)), 1) if errors else None,
        "median_px": round(float(np.median(errors)), 1) if errors else None,
        "p90_px": round(float(np.percentile(errors, 90)), 1) if errors else None,
        "max_px": round(float(np.max(errors)), 1) if errors else None,
        "mean_inference_ms": round(float(np.mean(latencies)), 1) if latencies else None,
        "images": rows_out,
    }


def montage(results, out_dir):
    gt = json.loads(GT.read_text(encoding="utf-8"))
    tiles = []
    for item in gt["images"]:
        frame = cv2.imread(str(ROOT / item["path"]))
        base = frame.copy()
        for b in item["boundaries"]:
            for r in b["rows"]:
                cv2.circle(base, (r["x"], r["y"]), 6, (0, 0, 255), -1)
                cv2.putText(base, f"{r['x']}", (r["x"] + 8, r["y"] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        panels = [("ground truth", base)]
        for res in results:
            if "error" in res:
                continue
            det = UltraFastLaneDetector(device="cpu", model_key=res["model"], preprocess=res["preprocess"])
            if not det.is_loaded:
                continue
            vis = frame.copy()
            lanes, _ = det.detect(frame)
            det.annotate_frame(vis, lanes)
            panels.append((f"{res['model']}/{res['preprocess']}", vis))
        h = min(420, base.shape[0])
        row = []
        for label, img in panels:
            s = h / img.shape[0]
            t = cv2.resize(img, (int(img.shape[1] * s), h))
            cv2.putText(t, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            row.append(t)
        tiles.append(np.hstack(row))
    w = max(t.shape[1] for t in tiles)
    padded = [cv2.copyMakeBorder(t, 0, 8, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
              for t in tiles]
    out = out_dir / "lane_compare.jpg"
    cv2.imwrite(str(out), np.vstack(padded))
    print(f"montage: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tusimple", "culane"])
    ap.add_argument("--preprocess", nargs="+", default=["stretch"])
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--gt", default=None,
                    help="ground-truth file (default labels/lane/gt.json; "
                         "use labels/lane/clips_gt.json for the clip frames)")
    args = ap.parse_args()
    global GT
    if args.gt:
        GT = Path(args.gt) if Path(args.gt).is_absolute() else ROOT / args.gt

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "runs" / f"lane_eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [run(m, p) for m in args.models for p in args.preprocess]
    for r in results:
        if "error" in r:
            print(f"{r['model']}/{r['preprocess']}: LOAD FAILED {r['error']}")
            continue
        print(f"\n== {r['model']} / {r['preprocess']} ==")
        print(f"   coverage {r['covered']}/{r['n_labelled_points']} = {r['coverage']}")
        print(f"   |error| mean {r['mae_px']} px, median {r['median_px']} px, "
              f"p90 {r['p90_px']} px, max {r['max_px']} px")
        print(f"   inference {r['mean_inference_ms']} ms (CPU)")
        for img in r["images"]:
            line = " ".join(f"{row['y']}:{row['model_x']}({row['error']})" for row in img["rows"])
            print(f"     {Path(img['path']).name:26s} lanes={img['n_lanes']}  {line}")

    if args.montage:
        montage(results, out_dir)

    (out_dir / "report.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data": str(GT.relative_to(ROOT)),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nreport: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
