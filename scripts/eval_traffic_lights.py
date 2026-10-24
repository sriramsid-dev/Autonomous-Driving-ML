"""
Traffic-light state evaluation on manually labelled real crops.

    python scripts/eval_traffic_lights.py                    # v1.4 on dev + held-out
    python scripts/eval_traffic_lights.py --compare          # v1.4 / v1.1 / v1.0
    python scripts/eval_traffic_lights.py --set v2 --compare  # held-out only

Two sets, never mixed:

  dev      labels/traffic_lights/gt.json     39 crops from 10 photos already in
           the repo. The v1.4 thresholds were chosen here, so its numbers are
           tuning results, not evidence of generalisation.

  v2       labels/traffic_lights/v2/gt.json  18 crops from 12 Wikimedia Commons
           photographs fetched after v1.4 was frozen. Never used to pick a
           threshold. This is the only number that speaks to generalisation,
           and it is n=9 vehicle signals - far too small for an accuracy claim.

Crops are single frames, so temporal voting is off (classify() is called without
a key). "unknown" is scored as an abstention, not as a wrong answer, and the
closed-set accuracy counts it as not-correct so the two cannot be confused.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SETS = {
    "dev": ROOT / "labels" / "traffic_lights" / "gt.json",
    "v2": ROOT / "labels" / "traffic_lights" / "v2" / "gt.json",
}
VEHICLE = ("red", "yellow", "green")
STATES = ["red", "yellow", "green", "unknown"]


def load_from_git(commit: str, module_rel: str, name: str):
    """Import a classifier straight out of a past commit so comparisons stay reproducible."""
    src = subprocess.run(
        ["git", "show", f"{commit}:{module_rel}"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / f"{name}.py"
    tmp.write_text(src, encoding="utf-8")
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TrafficLightClassifier()


def classifiers(which):
    from modules.traffic_light_classification import TrafficLightClassifier

    out = [("v1.4 per-state bezel contrast", TrafficLightClassifier())]
    if "v11" in which:
        out.append(("v1.1 global-mask circular hue", load_from_git(
            "6c79bbe", "modules/traffic_light_classification.py", "tl_v11")))
    if "v10" in which:
        out.append(("v1.0 HSV + spatial prior", load_from_git(
            "24c9cd0", "modules/traffic_light_classification.py", "tl_v10")))
    return out


def confusion(pairs):
    m = {g: {p: 0 for p in STATES} for g in VEHICLE}
    for g, p in pairs:
        m[g][p if p in STATES else "unknown"] += 1
    return m


def prf(pairs, label):
    tp = sum(1 for g, p in pairs if g == label and p == label)
    fp = sum(1 for g, p in pairs if g != label and p == label)
    fn = sum(1 for g, p in pairs if g == label and p != label)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    return {"support": tp + fn, "precision": prec, "recall": rec, "f1": f1}


def bucket(rows, label):
    sub = [r for r in rows if r["label"] == label]
    if not sub:
        return None
    abst = sum(1 for r in sub if r["pred"] == "unknown")
    named = [r["pred"] for r in sub if r["pred"] != "unknown"]
    return {
        "n": len(sub),
        "abstained": abst,
        "abstention_rate": round(abst / len(sub), 3),
        "predicted": dict((k, named.count(k)) for k in sorted(set(named))) or {},
    }


def evaluate(clf, samples, tag, set_name):
    rows = []
    for s in samples:
        crop = cv2.imread(str(ROOT / s["crop"]))
        state, conf = clf.classify(crop) if crop is not None else ("unknown", 0.0)
        rows.append({**s, "pred": state, "pred_conf": round(float(conf), 3)})

    veh = [r for r in rows if r["label"] in VEHICLE]
    pairs = [(r["label"], r["pred"]) for r in veh]
    correct = sum(1 for g, p in pairs if g == p)
    abst = sum(1 for _, p in pairs if p == "unknown")
    wrong = len(pairs) - correct - abst

    result = {
        "set": set_name,
        "classifier": tag,
        "n_crops": len(rows),
        "vehicle_signals": {
            "n": len(veh),
            "correct": correct,
            "wrong": wrong,
            "abstained": abst,
            "accuracy_closed_set": round(correct / len(veh), 3) if veh else None,
            "accuracy_when_committed": round(correct / (correct + wrong), 3) if (correct + wrong) else None,
            "abstention_rate": round(abst / len(veh), 3) if veh else None,
            "confusion_matrix": confusion(pairs),
            "per_class": {c: prf(pairs, c) for c in VEHICLE},
            "yellow_support_note": (
                "no verified amber sample in the dev set, so no amber metric is reported there"
                if set_name == "dev" else None
            ),
        },
        "unlit_housings": bucket(rows, "unlit"),
        "not_a_traffic_light": bucket(rows, "not_a_traffic_light"),
        "other_signals": bucket(rows, "other_signal"),
        "ambiguous": bucket(rows, "ambiguous"),
        "labels_missed_by_detector": None,
        "samples": rows,
    }
    missed = [r for r in veh if r.get("crop_missed_lamp")]
    if missed:
        result["labels_missed_by_detector"] = {
            "n": len(missed),
            "abstained": sum(1 for r in missed if r["pred"] == "unknown"),
            "note": "crops where the detector box missed the lit lamp; abstention is the desired outcome",
        }
    return result


def show(r):
    v = r["vehicle_signals"]
    print(f"\n== {r['classifier']}  [{r['set']}] ==")
    if not v["n"]:
        print("   no vehicle signals in this set")
        return
    print(f"   n={v['n']}  correct={v['correct']}  wrong={v['wrong']}  abstained={v['abstained']}")
    print(f"   closed-set accuracy={v['accuracy_closed_set']}  "
          f"accuracy when committed={v['accuracy_when_committed']}  abstention={v['abstention_rate']}")
    for c in VEHICLE:
        p = v["per_class"][c]
        if not p["support"]:
            print(f"   {c:6s} support=0  (no metric reported)")
            continue
        fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
        print(f"   {c:6s} support={p['support']:>2}  P={fmt(p['precision'])}  R={fmt(p['recall'])}  F1={fmt(p['f1'])}")
    print(f"   confusion (rows=truth, cols={STATES}):")
    for g in VEHICLE:
        if any(v["confusion_matrix"][g].values()):
            print(f"     {g:6s} {v['confusion_matrix'][g]}")
    for key in ("unlit_housings", "not_a_traffic_light", "other_signals", "ambiguous"):
        b = r[key]
        if b:
            print(f"   {key}: n={b['n']} abstained={b['abstained']} ({b['abstention_rate']}) "
                  f"predicted={b['predicted']}")
    if r["labels_missed_by_detector"]:
        m = r["labels_missed_by_detector"]
        print(f"   crops where the box missed the lit lamp: n={m['n']} abstained={m['abstained']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS) + ["all"], default="all")
    ap.add_argument("--compare", action="store_true", help="also run v1.1 and v1.0 from git")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    which = ["v14", "v11", "v10"] if args.compare else ["v14"]
    names = SETS if args.set == "all" else {args.set: SETS[args.set]}

    results = []
    for set_name, path in names.items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for tag, clf in classifiers(which):
            results.append(evaluate(clf, doc["samples"], tag, set_name))

    for r in results:
        show(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "outputs" / "runs" / f"tl_eval_{stamp}" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "note": "dev = tuning set, v2 = held-out. unknown is scored as abstention.",
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
