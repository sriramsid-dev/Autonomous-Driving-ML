"""
Tracking-continuity proxy comparison (class-filtered vs all-class tracking).

    python scripts/eval_tracking.py --video "D:/downloads/Wyamo Car video.mp4" --frames 120

There is NO ground-truth track annotation for this footage, so MOTA, IDF1 and
the true number of ID switches CANNOT be computed and are not reported.

What is measured instead, and how to read it:
  For each pair of consecutive frames, detections are matched by IoU >= 0.5
  within the same class (spatial correspondence only, no identity prior).
  * continuity_rate = share of matched pairs that keep the same track id.
    Higher is better: the tracker did not rename an object it kept following.
  * reassigned_pairs = matched pairs whose id changed. Without GT these are
    *pseudo* switches: a real switch and a legitimate new object at the same
    place are indistinguishable, so this is a proxy, not an ID-switch count.
  * ids_total / median_track_length describe churn: fewer ids for the same
    number of matched pairs means fewer track fragments.

Both settings use identical frames, conf, IoU and imgsz; only the class filter
differs (filter_classes=True is the v1.1 behaviour).
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modules.object_detection_tracking import ObjectDetectionTracker  # noqa: E402


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run(video, n_frames, filter_classes, device, imgsz):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    tracker = ObjectDetectionTracker(device=device, imgsz=imgsz)
    tracker.reset_tracker()

    frames_out, times = [], []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        objs, ms = tracker.track(frame, persist=True, filter_classes=filter_classes)
        times.append(ms)
        frames_out.append([{"box": o["box"], "class_id": o["class_id"], "track_id": o["track_id"]}
                           for o in objs if o["track_id"] is not None])
    cap.release()

    matched = same_id = 0
    ids = set()
    lengths = {}
    for dets in frames_out:
        for d in dets:
            ids.add(d["track_id"])
            lengths[d["track_id"]] = lengths.get(d["track_id"], 0) + 1

    for prev, cur in zip(frames_out, frames_out[1:]):
        used = set()
        for d in cur:
            best, best_iou = None, 0.5
            for j, p in enumerate(prev):
                if j in used or p["class_id"] != d["class_id"]:
                    continue
                v = iou(d["box"], p["box"])
                if v >= best_iou:
                    best, best_iou = j, v
            if best is not None:
                used.add(best)
                matched += 1
                if prev[best]["track_id"] == d["track_id"]:
                    same_id += 1

    return {
        "setting": "class-filtered" if filter_classes else "all-classes (pre-v1.1)",
        "frames": len(frames_out),
        "detections": int(sum(len(f) for f in frames_out)),
        "ids_total": len(ids),
        "median_track_length_frames": statistics.median(lengths.values()) if lengths else 0,
        "matched_pairs": matched,
        "continuity_rate": round(same_id / matched, 4) if matched else None,
        "reassigned_pairs": matched - same_id,
        "latency_ms": {
            "mean": round(statistics.mean(times), 1) if times else None,
            "p50": round(statistics.median(times), 1) if times else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=config.DETECTION_IMGSZ)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "runs" / f"{run_id}_tracking"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for filtered in (True, False):
        r = run(args.video, args.frames, filtered, args.device, args.imgsz)
        results.append(r)
        print(f"{r['setting']:26s} frames={r['frames']:3d} dets={r['detections']:4d} "
              f"ids={r['ids_total']:3d} matched={r['matched_pairs']:4d} "
              f"continuity={r['continuity_rate']} p50={r['latency_ms']['p50']}ms")

    report = {
        "run_id": run_id,
        "video": args.video,
        "frames_requested": args.frames,
        "imgsz": args.imgsz,
        "caveat": ("No ground-truth tracks exist for this footage. MOTA/IDF1 and true ID-switch "
                   "counts cannot and are not computed. The figures above are IoU-correspondence "
                   "proxies for continuity, not tracking accuracy."),
        "results": results,
    }
    out = Path(args.out) if args.out else run_dir / "tracking_proxy.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
