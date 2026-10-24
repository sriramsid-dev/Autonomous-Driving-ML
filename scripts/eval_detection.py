"""
Object-detection evaluation on the verified local label set.

    python scripts/eval_detection.py --models yolo11n.pt yolo11s.pt
    python scripts/eval_detection.py --models yolo11n.pt yolo11s.pt --imgsz 960
    python scripts/eval_detection.py --models yolo11m.pt --imgsz 640

Protocol (fixed for every model in a run):
  * only images whose GT file has status == "verified" are used;
  * the same images, the same confidence floor, NMS IoU, input size and device;
  * AP is all-point interpolated per class at IoU 0.5, greedy matching per image;
  * precision/recall are reported at the deployment operating point (conf 0.25);
  * classes with fewer than MIN_SUPPORT instances are flagged as insufficient;
  * a paired bootstrap over images gives a 95% interval for both the AP and the
    AP difference between the first model and each other model.

Nothing here is a COCO/custom-benchmark score: the label set is small and
proposal-derived (see labels/detection/gt/*.json for per-image provenance).
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from utils.evaluation import _ap_single_class  # noqa: E402

GT_DIR = ROOT / "labels" / "detection" / "gt"
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MIN_SUPPORT = 5
BOOTSTRAP_N = 1000


def load_gt():
    images = []
    for f in sorted(GT_DIR.glob("*.json")):
        doc = json.loads(f.read_text())
        if doc.get("status") != "verified":
            continue
        if doc.get("split") != "eval":
            continue
        gt = [{"box": b["box"], "class_id": int(b["class_id"])} for b in doc["boxes"]]
        images.append({"id": doc["image_id"], "path": doc["path"], "gt": gt, "size": doc["size"]})
    return images


def predict_all(model, images, imgsz, conf_floor, iou, device):
    # Warm up once so the first timed image does not absorb model/graph setup.
    model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz,
                  device=device, verbose=False)
    preds, timings = [], []
    for img in images:
        frame = cv2.imread(str(ROOT / img["path"]))
        if frame is None:
            raise SystemExit(f"unreadable image: {img['path']}")
        t0 = time.perf_counter()
        res = model.predict(frame, conf=conf_floor, iou=iou, imgsz=imgsz,
                            classes=sorted(CLASS_NAMES), device=device, verbose=False)[0]
        timings.append((time.perf_counter() - t0) * 1000.0)
        out = []
        if res.boxes is not None:
            for box, c, k in zip(res.boxes.xyxy.cpu().numpy(),
                                 res.boxes.conf.cpu().numpy(),
                                 res.boxes.cls.int().cpu().numpy()):
                out.append({"box": [float(v) for v in box], "class_id": int(k), "confidence": float(c)})
        preds.append(out)
    return preds, timings


def ap_per_class(preds, gts, class_ids, iou_thresh=0.5):
    out = {}
    for cid in class_ids:
        p = [[d for d in per_img if d["class_id"] == cid] for per_img in preds]
        g = [[d for d in per_img if d["class_id"] == cid] for per_img in gts]
        out[cid] = _ap_single_class(p, g, iou_thresh)
    return out


def operating_point(preds, gts, class_ids, conf_thr, iou_thresh=0.5):
    tp = {c: 0 for c in class_ids}
    fp = {c: 0 for c in class_ids}
    fn = {c: 0 for c in class_ids}
    for per_img_p, per_img_g in zip(preds, gts):
        for cid in class_ids:
            ps = sorted([d for d in per_img_p if d["class_id"] == cid and d["confidence"] >= conf_thr],
                        key=lambda d: -d["confidence"])
            gs = [d for d in per_img_g if d["class_id"] == cid]
            matched = [False] * len(gs)
            for p in ps:
                best, best_iou = -1, iou_thresh
                for i, g in enumerate(gs):
                    if matched[i]:
                        continue
                    iou = _iou(p["box"], g["box"])
                    if iou >= best_iou:
                        best, best_iou = i, iou
                if best >= 0:
                    matched[best] = True
                    tp[cid] += 1
                else:
                    fp[cid] += 1
            fn[cid] += matched.count(False)
    rows = {}
    for cid in class_ids:
        prec = tp[cid] / (tp[cid] + fp[cid]) if (tp[cid] + fp[cid]) else None
        rec = tp[cid] / (tp[cid] + fn[cid]) if (tp[cid] + fn[cid]) else None
        rows[cid] = {"tp": tp[cid], "fp": fp[cid], "fn": fn[cid],
                     "precision": prec, "recall": rec}
    return rows


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def bootstrap_delta(preds_a, preds_b, gts, class_ids, rng, n=BOOTSTRAP_N):
    """Paired bootstrap over images: AP(A) - AP(B) for the pooled mAP."""
    idx_all = np.arange(len(images_global))
    diffs = []
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        pa = [preds_a[i] for i in idx]
        pb = [preds_b[i] for i in idx]
        gg = [gts[i] for i in idx]
        aps_a, aps_b = [], []
        for cid in class_ids:
            ap_a = _ap_single_class([[d for d in x if d["class_id"] == cid] for x in pa],
                                    [[d for d in x if d["class_id"] == cid] for x in gg], 0.5)
            ap_b = _ap_single_class([[d for d in x if d["class_id"] == cid] for x in pb],
                                    [[d for d in x if d["class_id"] == cid] for x in gg], 0.5)
            if ap_a["total_gt"] > 0:
                aps_a.append(ap_a["ap50"])
                aps_b.append(ap_b["ap50"])
        diffs.append(float(np.mean(aps_a) - np.mean(aps_b)))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(np.mean(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    ap.add_argument("--imgsz", type=int, default=config.DETECTION_IMGSZ)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--conf-operating", type=float, default=config.DETECTION_CONF_THRESHOLD)
    ap.add_argument("--iou", type=float, default=config.DETECTION_IOU_THRESHOLD)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    global images_global
    images_global = load_gt()
    if not images_global:
        raise SystemExit("no verified eval labels found; run scripts/annotate.py first")
    gts = [img["gt"] for img in images_global]

    class_support = {cid: sum(1 for g in gts for d in g if d["class_id"] == cid) for cid in CLASS_NAMES}
    evaluable = [cid for cid, n in class_support.items() if n > 0]
    if not evaluable:
        raise SystemExit("no GT instances")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "runs" / f"{run_id}_detect_imgsz{args.imgsz}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"images: {len(images_global)} | instances: {sum(class_support.values())} | imgsz={args.imgsz}")
    print("support:", {CLASS_NAMES[c]: n for c, n in class_support.items()})

    results, raw_preds, raw_times = {}, {}, {}
    for name in args.models:
        model = YOLO(str(ROOT / "models" / name))
        preds, times = predict_all(model, images_global, args.imgsz, args.conf_floor, args.iou, args.device)
        aps = ap_per_class(preds, gts, evaluable, args.iou)
        op = operating_point(preds, gts, evaluable, args.conf_operating, args.iou)
        pooled = float(np.mean([aps[c]["ap50"] for c in evaluable]))
        results[name] = {
            "map50_iou50": round(pooled, 4),
            "per_class": {
                CLASS_NAMES[c]: {
                    "ap50": round(aps[c]["ap50"], 4),
                    "precision_at_conf": _r(op[c]["precision"]),
                    "recall_at_conf": _r(op[c]["recall"]),
                    "tp": op[c]["tp"], "fp": op[c]["fp"], "fn": op[c]["fn"],
                    "support": class_support[c],
                    "insufficient_support": class_support[c] < MIN_SUPPORT,
                }
                for c in evaluable
            },
            "latency_ms": {
                "mean": round(statistics.mean(times), 1),
                "p50": round(statistics.median(times), 1),
                "p90": round(float(np.percentile(times, 90)), 1),
            },
        }
        raw_preds[name] = preds
        raw_times[name] = times
        print(f"  {name}: mAP@0.5={pooled:.3f}  p50={results[name]['latency_ms']['p50']}ms")

    # paired bootstrap of the AP difference, first model as the reference
    rng = np.random.default_rng(0)
    base = args.models[0]
    for name in args.models[1:]:
        lo, hi, mean = bootstrap_delta(raw_preds[base], raw_preds[name], gts, evaluable, rng)
        results[name]["delta_vs_" + Path(base).stem] = {
            "mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0),
        }

    report = {
        "run_id": run_id,
        "protocol": {
            "imgsz": args.imgsz, "conf_floor": args.conf_floor,
            "conf_operating": args.conf_operating, "nms_iou": args.iou,
            "ap": "all-point interpolated per class, IoU 0.5, greedy per-image matching",
            "bootstrap": f"paired over images, {BOOTSTRAP_N} resamples, seed 0",
            "device": args.device,
        },
        "env": {"python": platform.python_version(), "platform": platform.platform()},
        "dataset": {
            "images": [im["id"] for im in images_global],
            "n_images": len(images_global),
            "n_instances": int(sum(class_support.values())),
            "support": {CLASS_NAMES[k]: v for k, v in class_support.items()},
            "label_tier": "verified-proposal (presence verified by eye, boxes not redrawn)",
        },
        "models": results,
    }
    out = Path(args.out) if args.out else run_dir / "report.json"
    out.write_text(json.dumps({**report, "per_image_predictions": {
        Path(m).stem: [len(p) for p in raw_preds[m]] for m in args.models}}, indent=2))

    print("\n| model | mAP@0.5 | car AP | person AP | truck AP | p50 ms |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        pc = r["per_class"]
        print(f"| {name} | {r['map50_iou50']:.3f} | {pc.get('car',{}).get('ap50','-')} | "
              f"{pc.get('person',{}).get('ap50','-')} | {pc.get('truck',{}).get('ap50','-')} | "
              f"{r['latency_ms']['p50']} |")
    print(f"\nreport: {out}")


def _r(v):
    return None if v is None else round(float(v), 3)


if __name__ == "__main__":
    main()
