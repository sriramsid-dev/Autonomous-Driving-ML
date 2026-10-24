"""
Reproducible evaluation harness for Version 1 perception modules.

Usage:
    python scripts/evaluate.py                # run every available check
    python scripts/evaluate.py --only tl,lane
    python scripts/evaluate.py --lane-preprocess letterbox

What it reports, and from what:
  * traffic-light state accuracy on `labels/eval_labels.json` (agent-labelled,
    tiny - see the provenance block; not a benchmark)
  * lane ego-boundary error at labelled rows (same caveat)
  * object detection: only if `labels.eval_labels.json:object_boxes` is
    populated; otherwise it reports that it CANNOT compute mAP and says so
  * runtime (median/p90 latency per stage) measured on this machine

Every result JSON records the environment and the exact configuration used, so
two runs are comparable and a run can be reproduced.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modules.lane_detection import UltraFastLaneDetector  # noqa: E402
from modules.object_detection_tracking import ObjectDetectionTracker  # noqa: E402
from modules.traffic_light_classification import TrafficLightClassifier  # noqa: E402
from modules.traffic_light_detection import TrafficLightDetector  # noqa: E402
from utils.evaluation import evaluate_tl_classification  # noqa: E402

LABELS_PATH = ROOT / "labels" / "eval_labels.json"


def _pct(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def evaluate_traffic_lights(labels, device):
    clf = TrafficLightClassifier()
    det = TrafficLightDetector(device=device)
    rows, preds, gts = [], [], []

    for case in labels["traffic_light_states"]:
        path = ROOT / case["path"]
        if not path.exists():
            rows.append({"path": case["path"], "status": "missing"})
            continue
        img = cv2.imread(str(path))
        if img is None:
            rows.append({"path": case["path"], "status": "unreadable"})
            continue

        t0 = time.perf_counter()
        if case["kind"] == "synthetic_crop":
            # Direct crop classification, no temporal voting (single sample).
            state, conf = clf.classify(img)
            detected = 1
        else:
            dets, _ = det.detect(img)
            states = []
            for d in dets:
                s, c = clf.classify(d["crop"])
                states.append((s, c))
            valid = [s for s in states if s[0] != "unknown"]
            state, conf = (max(valid, key=lambda x: x[1]) if valid else (states[0] if states else ("unknown", 0.0)))
            detected = len(dets)
        dt = (time.perf_counter() - t0) * 1000.0

        preds.append(state)
        gts.append(case["state"])
        rows.append(
            {
                "path": case["path"],
                "kind": case["kind"],
                "ground_truth": case["state"],
                "predicted": state,
                "confidence": round(float(conf), 3),
                "lights_detected": detected,
                "latency_ms": round(dt, 1),
                "correct": state == case["state"],
            }
        )

    natural = [r for r in rows if r.get("kind") in ("scene", "natural")]
    synthetic = [r for r in rows if r.get("kind") == "synthetic_crop"]
    return {
        "samples": rows,
        "task": "detect-then-classify (scene) or classify-only (crop)",
        "metrics_natural_only": {
            "n": len(natural),
            "correct": sum(1 for r in natural if r.get("correct")),
            "accuracy": round(sum(1 for r in natural if r.get("correct")) / len(natural), 3) if natural else None,
        },
        "metrics_synthetic_crops_only": {
            "n": len(synthetic),
            "correct": sum(1 for r in synthetic if r.get("correct")),
            "accuracy": round(sum(1 for r in synthetic if r.get("correct")) / len(synthetic), 3) if synthetic else None,
            "note": "matplotlib renderings, not sensor data; kept only for regression continuity",
        },
        "all_samples_unchanged_api": evaluate_tl_classification(preds, gts),
    }


def evaluate_lanes(labels, device, preprocess):
    det = UltraFastLaneDetector(device=device, preprocess=preprocess)
    if not det.is_loaded:
        return {"error": det.load_error or "lane model not loaded"}

    out = []
    for case in labels["lane_ego_boundaries"]:
        path = ROOT / case["path"]
        img = cv2.imread(str(path))
        if img is None:
            out.append({"path": case["path"], "status": "unreadable"})
            continue
        t0 = time.perf_counter()
        lanes, _ = det.detect(img)
        dt = (time.perf_counter() - t0) * 1000.0
        det._last_shape = img.shape[:2]
        left, right, diag = det._select_ego_boundaries(lanes)

        errs = []
        if left is not None and right is not None:
            for row in case["rows"]:
                ly = np.array([p[1] for p in left], dtype=np.float32)
                ry = np.array([p[1] for p in right], dtype=np.float32)
                if ly.min() <= row["y"] <= ly.max() and ry.min() <= row["y"] <= ry.max():
                    lx = float(np.interp(row["y"], ly, np.array([p[0] for p in left], dtype=np.float32)))
                    rx = float(np.interp(row["y"], ry, np.array([p[0] for p in right], dtype=np.float32)))
                    errs.append(
                        {
                            "y": row["y"],
                            "left_error_px": round(abs(lx - row["left_x"]), 1),
                            "right_error_px": round(abs(rx - row["right_x"]), 1),
                        }
                    )

        corridor = det.annotate_frame(img.copy(), lanes, draw_drive_area=True)
        out.append(
            {
                "path": case["path"],
                "lane_points": [len(l) for l in lanes],
                "corridor_established": det.last_corridor_established,
                "diagnostics": diag,
                "row_errors": errs,
                "mean_abs_error_px": round(statistics.mean([e["left_error_px"] for e in errs] + [e["right_error_px"] for e in errs]), 1) if errs else None,
                "latency_ms": round(dt, 1),
            }
        )

    errs = [e["left_error_px"] for r in out for e in r.get("row_errors", [])] + \
           [e["right_error_px"] for r in out for e in r.get("row_errors", [])]
    return {
        "preprocess": preprocess,
        "images": out,
        "mean_abs_boundary_error_px": round(statistics.mean(errs), 1) if errs else None,
        "images_with_corridor": sum(1 for r in out if r.get("corridor_established")),
        "images_evaluated": len(out),
    }


def evaluate_objects(labels, device):
    boxes = labels.get("object_boxes") or []
    if not boxes:
        return {
            "status": "not evaluated",
            "reason": "no bounding-box labels exist in labels/eval_labels.json",
            "implication": "No mAP/precision/recall is reported for Module A for this run.",
        }
    tracker = ObjectDetectionTracker(device=device)
    preds, gts, lat = [], [], []
    for case in boxes:
        img = cv2.imread(str(ROOT / case["path"]))
        if img is None:
            continue
        t0 = time.perf_counter()
        preds.append(tracker.detect(img))
        lat.append((time.perf_counter() - t0) * 1000.0)
        gts.append(case["boxes"])
    from utils.evaluation import evaluate_detection_map50

    res = evaluate_detection_map50(preds, gts, iou_thresh=config.IOU_THRESHOLD_EVAL)
    res["latency_p50_ms"] = round(statistics.median(lat), 1) if lat else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="tl,lane,objects")
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--lane-preprocess", default=config.LANE_PREPROCESS, choices=["stretch", "letterbox", "crop"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not LABELS_PATH.exists():
        raise SystemExit(f"label file missing: {LABELS_PATH}")
    labels = json.loads(LABELS_PATH.read_text())
    wanted = {t.strip() for t in args.only.split(",") if t.strip()}

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": {
            "file": str(LABELS_PATH.relative_to(ROOT)),
            "provenance": labels.get("provenance", {}),
        },
        "env": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch": _safe_version("torch"),
            "ultralytics": _safe_version("ultralytics"),
            "opencv": cv2.__version__,
            "device": args.device,
        },
        "config": {
            "detection_imgsz": config.DETECTION_IMGSZ,
            "tl_detection_imgsz": config.TL_DETECTION_IMGSZ,
            "dt_conf": config.DETECTION_CONF_THRESHOLD,
            "tl_conf": config.TL_DETECTION_CONF_THRESHOLD,
            "lane_preprocess": args.lane_preprocess,
            "lane_temporal": config.LANE_TEMPORAL_SMOOTHING,
            "lane_existence_threshold": config.LANE_EXISTENCE_THRESHOLD,
            "tl_min_state_conf": config.TL_MIN_STATE_CONF,
        },
    }

    if "tl" in wanted:
        report["traffic_lights"] = evaluate_traffic_lights(labels, args.device)
    if "lane" in wanted:
        report["lanes"] = evaluate_lanes(labels, args.device, args.lane_preprocess)
    if "objects" in wanted:
        report["objects"] = evaluate_objects(labels, args.device)

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else config.METRICS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k not in ("env", "labels", "config")}, indent=2)[:4000])
    print(f"\n[evaluate] full report: {out}")


def _safe_version(mod: str):
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return "unavailable"


if __name__ == "__main__":
    main()
