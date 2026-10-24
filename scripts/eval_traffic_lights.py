"""
Held-out traffic-light state evaluation on manually labelled real crops.

    python scripts/eval_traffic_lights.py
    python scripts/eval_traffic_lights.py --compare-legacy   # also runs the v1.0 heuristic from git

Data: labels/traffic_lights/gt.json — 39 crops cut from 10 real photographs,
each inspected and labelled by eye (see the provenance block in that file).
Crops are single frames, so temporal voting is disabled here by calling
classify() without a key: this measures the per-crop classifier only.

Reported:
  * vehicle-signal subset (red/green): accuracy, confusion matrix, per-class
    precision/recall/F1, and the abstention rate ("unknown" outputs);
  * unlit housings: abstention rate (a good reject option abstains);
  * detector false positives among the boxes the detector called "traffic light";
  * breakdown by lighting condition and by crop size.
Yellow has no support in this set; nothing is reported for it.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.traffic_light_classification import TrafficLightClassifier  # noqa: E402

GT_PATH = ROOT / "labels" / "traffic_lights" / "gt.json"
STATES = ["red", "yellow", "green", "unknown"]


def load_legacy():
    """Load the pre-v1.1 heuristic straight from the baseline commit."""
    src = subprocess.run(
        ["git", "show", "24c9cd0:modules/traffic_light_classification.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "legacy_tl.py"
    tmp.write_text(src, encoding="utf-8")
    import importlib.util

    spec = importlib.util.spec_from_file_location("legacy_tl", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TrafficLightClassifier()


def confusion(pairs, classes):
    m = {g: {p: 0 for p in classes} for g in classes}
    for g, p in pairs:
        m[g][p] += 1
    return m


def prf(pairs, label):
    tp = sum(1 for g, p in pairs if g == label and p == label)
    fp = sum(1 for g, p in pairs if g != label and p == label)
    fn = sum(1 for g, p in pairs if g == label and p != label)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    return {"support": tp + fn, "precision": prec, "recall": rec, "f1": f1}


def evaluate(clf, samples, tag):
    rows = []
    for s in samples:
        crop = cv2.imread(str(ROOT / s["crop"]))
        state, conf = clf.classify(crop) if crop is not None else ("unknown", 0.0)
        rows.append({**s, "pred": state, "pred_conf": round(float(conf), 3)})

    vehicle = [r for r in rows if r["label"] in ("red", "green")]
    pairs = [(r["label"], r["pred"]) for r in vehicle]
    cm = confusion(pairs, STATES)
    correct = sum(1 for g, p in pairs if g == p)
    abstained = sum(1 for _, p in pairs if p == "unknown")

    unlit = [r for r in rows if r["label"] == "unlit"]
    other = [r for r in rows if r["label"] == "other_signal"]
    notlight = [r for r in rows if r["label"] == "not_a_traffic_light"]

    def by(pred_rows, cond):
        rows_c = [r for r in pred_rows if r["condition"] == cond]
        if not rows_c:
            return None
        ok = sum(1 for r in rows_c if r["pred"] == r["label"])
        return {"n": len(rows_c), "correct": ok, "accuracy": round(ok / len(rows_c), 3)}

    return {
        "classifier": tag,
        "n_crops": len(rows),
        "vehicle_signal_subset": {
            "n": len(vehicle),
            "correct": correct,
            "accuracy": round(correct / len(vehicle), 3) if vehicle else None,
            "abstentions": abstained,
            "abstention_rate": round(abstained / len(vehicle), 3) if vehicle else None,
            "confusion_matrix": cm,
            "per_class": {c: prf(pairs, c) for c in ("red", "green")},
            "by_condition": {c: by(vehicle, c) for c in ("day", "night", "glare")},
            "by_crop_size": {
                "small_<=24px_wide": {
                    "n": sum(1 for r in vehicle if r["w"] <= 24),
                    "correct": sum(1 for r in vehicle if r["w"] <= 24 and r["pred"] == r["label"]),
                }
            },
        },
        "unlit_housings": {
            "n": len(unlit),
            "abstained": sum(1 for r in unlit if r["pred"] == "unknown"),
            "abstention_rate": round(sum(1 for r in unlit if r["pred"] == "unknown") / len(unlit), 3) if unlit else None,
        },
        "other_signals": {"n": len(other), "predictions": [(r["pred"], r["pred_conf"]) for r in other]},
        "detector_false_positives": {
            "n": len(notlight), "of_boxes": len(rows),
            "rate": round(len(notlight) / len(rows), 3) if rows else None,
            "details": [(r["src"].split("/")[-1], r["w"], r["h"], r["pred"]) for r in notlight],
        },
        "samples": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-legacy", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    doc = json.loads(GT_PATH.read_text())
    samples = doc["samples"]

    results = [evaluate(TrafficLightClassifier(), samples, "v1.1 blob+circular-hue")]
    if args.compare_legacy:
        results.append(evaluate(load_legacy(), samples, "v1.0 HSV+spatial-prior (baseline commit)"))

    for r in results:
        v = r["vehicle_signal_subset"]
        print(f"\n== {r['classifier']} ==")
        print(f" vehicle signals: {v['correct']}/{v['n']} correct, accuracy={v['accuracy']}, "
              f"abstention={v['abstention_rate']}")
        for c in ("red", "green"):
            print(f"   {c:6s} {v['per_class'][c]}")
        print(f"   unlit abstention: {r['unlit_housings']['abstained']}/{r['unlit_housings']['n']}")
        print(f"   detector FP: {r['detector_false_positives']['n']}/{r['detector_false_positives']['of_boxes']}")
        print(f"   confusion: {v['confusion_matrix']}")

    out = Path(args.out) if args.out else ROOT / "outputs" / "runs" / "traffic_lights" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"data": str(GT_PATH.relative_to(ROOT)), "results": results}, indent=2))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
