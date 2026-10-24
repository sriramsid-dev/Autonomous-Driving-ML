"""
Lightweight bounding-box labelling workflow (free, no external services).

Three steps, each a separate command so nothing is ever silently accepted:

  1. propose   run detectors at a low confidence and store *suggestions*
               (explicitly marked unverified) in labels/detection/proposals/
  2. render    write a numbered, gridded review image per frame so a human can
               accept / reject / add boxes by eye
  3. finalize  write the verified ground truth to labels/detection/gt/

A file in labels/detection/gt/ is the ONLY thing the evaluation scripts will
read, and it records who verified it and how. Proposals are never promoted
automatically.

Examples
--------
    python scripts/annotate.py init-manifest
    python scripts/annotate.py propose --split eval
    python scripts/annotate.py render --image-id my_test_images__sir_image
    python scripts/annotate.py finalize --image-id my_test_images__sir_image \
        --accept all --reject 3,7 --add "120,340,260,470,car"
    python scripts/annotate.py stats
    python scripts/annotate.py export-yolo --split eval

Class ids follow COCO (the detector's native labels), so no remapping is needed:
0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

LABEL_DIR = ROOT / "labels" / "detection"
MANIFEST = LABEL_DIR / "manifest.json"
PROP_DIR = LABEL_DIR / "proposals"
GT_DIR = LABEL_DIR / "gt"
REVIEW_DIR = ROOT / "outputs" / "runs" / "labelling_review"

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLOURS = {
    0: (0, 165, 255),
    1: (255, 0, 255),
    2: (0, 220, 0),
    3: (255, 0, 128),
    5: (0, 215, 255),
    7: (255, 140, 0),
}

# Scene groups: every group is one physical source, so a group can be assigned
# wholesale to either tuning ("dev") or evaluation, never split across both.
SCENES = {
    "variant_clip": {"split": "dev", "note": "480x848 portrait dashcam clip; frames also used to sample the video pipeline"},
    "waymo_still": {"split": "dev", "note": "Waymo Jaguar close-up still"},
    "street_stock": {"split": "eval", "note": "single-frame stock photograph, own scene"},
    "signal_scene": {"split": "eval", "note": "traffic-light street scene"},
    "highway": {"split": "eval", "note": "open highway, sparse traffic"},
    "indoor_persons": {"split": "eval", "note": "off-domain indoor photo; pedestrian robustness only"},
}


def _slug(path: Path) -> str:
    return f"{path.parent.name}__{path.stem}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------- manifest
def init_manifest() -> dict:
    entries = []
    stock = sorted((ROOT / "my_test_images").glob("sir_image*"))
    for p in stock:
        entries.append({"id": _slug(p), "path": str(p.relative_to(ROOT)), "scene": "street_stock"})
    for name, scene in [
        ("sample_wyamo.jpg", "waymo_still"),
        ("highway_road.jpg", "highway"),
        ("red_tl.jpg", "signal_scene"),
        ("green_tl.jpg", "signal_scene"),
        ("sample_gemini.jpg", "indoor_persons"),
    ]:
        p = ROOT / "test_data" / name
        if p.exists():
            entries.append({"id": _slug(p), "path": str(p.relative_to(ROOT)), "scene": scene})
    for p in sorted((ROOT / "test_data").glob("wyamo_f*.jpg")):
        entries.append({"id": _slug(p), "path": str(p.relative_to(ROOT)), "scene": "variant_clip"})

    for e in entries:
        e["split"] = SCENES[e["scene"]]["split"]
        img = cv2.imread(str(ROOT / e["path"]))
        e["size"] = [int(img.shape[1]), int(img.shape[0])] if img is not None else None

    manifest = {
        "created": _now(),
        "class_map": {str(k): v for k, v in CLASS_NAMES.items()},
        "scenes": SCENES,
        "images": entries,
    }
    for d in (LABEL_DIR, PROP_DIR, GT_DIR, REVIEW_DIR):
        d.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"manifest written: {MANIFEST} ({len(entries)} images, "
          f"{sum(1 for e in entries if e['split']=='eval')} eval / "
          f"{sum(1 for e in entries if e['split']=='dev')} dev)")
    return manifest


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return init_manifest()
    return json.loads(MANIFEST.read_text())


# -------------------------------------------------------------------- proposals
def _nms(boxes, scores, iou_thr=0.5):
    order = np.argsort(-np.asarray(scores))
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([_iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= iou_thr]
    return keep


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cmd_propose(args) -> int:
    from ultralytics import YOLO

    manifest = load_manifest()
    targets = [e for e in manifest["images"] if args.split in ("all", e["split"])]
    if not targets:
        print("no images matched")
        return 1

    models = []
    for name in args.models:
        path = ROOT / "models" / name
        models.append((name, YOLO(str(path))))

    PROP_DIR.mkdir(parents=True, exist_ok=True)
    for entry in targets:
        img = cv2.imread(str(ROOT / entry["path"]))
        if img is None:
            print(f"[skip] unreadable: {entry['path']}")
            continue
        proposals = []
        for name, model in models:
            res = model.predict(
                img, conf=args.conf, imgsz=args.imgsz, classes=sorted(CLASS_NAMES),
                device=args.device, verbose=False,
            )[0]
            if res.boxes is None:
                continue
            for box, conf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                      res.boxes.conf.cpu().numpy(),
                                      res.boxes.cls.int().cpu().numpy()):
                proposals.append({
                    "box": [round(float(v), 1) for v in box],
                    "class_id": int(cls),
                    "confidence": round(float(conf), 4),
                    "model": name,
                })

        # Merge duplicates from the two detectors, keeping the higher score.
        proposals.sort(key=lambda p: -p["confidence"])
        kept = []
        for p in proposals:
            if any(k["class_id"] == p["class_id"] and _iou(p["box"], k["box"]) > 0.6 for k in kept):
                continue
            kept.append(p)

        out = {
            "image_id": entry["id"],
            "path": entry["path"],
            "size": entry["size"],
            "verified": False,
            "warning": "MODEL PROPOSALS - not ground truth. Must be reviewed before use.",
            "conf_threshold": args.conf,
            "imgsz": args.imgsz,
            "models": [n for n, _ in models],
            "generated": _now(),
            "proposals": kept,
        }
        (PROP_DIR / f"{entry['id']}.json").write_text(json.dumps(out, indent=2))
        print(f"[propose] {entry['id']}: {len(kept)} suggestions")
    return 0


# ----------------------------------------------------------------------- render
def cmd_render(args) -> int:
    manifest = load_manifest()
    by_id = {e["id"]: e for e in manifest["images"]}
    ids = list(by_id) if args.image_id in (None, "all") else [args.image_id]
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for image_id in ids:
        if image_id not in by_id:
            print(f"unknown image id: {image_id}")
            continue
        entry = by_id[image_id]
        img = cv2.imread(str(ROOT / entry["path"]))
        if img is None:
            continue
        canvas = img.copy()
        gt_file = GT_DIR / f"{image_id}.json"
        prop_file = PROP_DIR / f"{image_id}.json"

        if gt_file.exists():
            boxes = json.loads(gt_file.read_text())["boxes"]
            title = "VERIFIED GT"
        elif prop_file.exists():
            boxes = json.loads(prop_file.read_text())["proposals"]
            title = "PROPOSALS (unverified)"
        else:
            boxes = []
            title = "no labels"

        # coordinate grid every 100 px keeps manual additions eyeball-accurate
        h, w = canvas.shape[:2]
        step = 100 if max(h, w) < 2000 else 200
        for x in range(0, w, step):
            cv2.line(canvas, (x, 0), (x, h), (60, 60, 60), 1)
            cv2.putText(canvas, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
        for y in range(0, h, step):
            cv2.line(canvas, (0, y), (w, y), (60, 60, 60), 1)
            cv2.putText(canvas, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)

        for i, b in enumerate(boxes, start=1):
            x1, y1, x2, y2 = [int(round(v)) for v in b["box"]]
            cid = int(b["class_id"])
            col = COLOURS.get(cid, (200, 200, 200))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
            conf = b.get("confidence")
            label = f"{i}:{CLASS_NAMES.get(cid, cid)}" + (f" {conf:.2f}" if conf is not None else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), col, -1)
            cv2.putText(canvas, label, (x1 + 2, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # keep the review image readable for large sources
        if max(h, w) > 1800:
            scale = 1800.0 / max(h, w)
            canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)))
        banner = np.zeros((34, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, f"{entry['id']}  |  {title}  |  {len(boxes)} boxes",
                    (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        out = np.vstack([banner, canvas])
        path = REVIEW_DIR / f"{image_id}{'_gt' if gt_file.exists() else ''}.jpg"
        cv2.imwrite(str(path), out)
        print(f"[render] {path.relative_to(ROOT)}  ({len(boxes)} boxes, {title})")
    return 0


# --------------------------------------------------------------------- finalize
def cmd_finalize(args) -> int:
    manifest = load_manifest()
    by_id = {e["id"]: e for e in manifest["images"]}
    entry = by_id.get(args.image_id)
    if entry is None:
        print(f"unknown image id: {args.image_id}")
        return 1
    prop_file = PROP_DIR / f"{args.image_id}.json"
    props = json.loads(prop_file.read_text())["proposals"] if prop_file.exists() else []

    reject = set()
    if args.reject:
        reject = {int(t) for t in args.reject.split(",") if t.strip()}

    if args.accept == "all":
        accepted_idx = [i for i in range(1, len(props) + 1) if i not in reject]
    elif args.accept in (None, "", "none"):
        accepted_idx = []
    else:
        accepted_idx = [int(t) for t in args.accept.split(",") if t.strip()]

    boxes, origins = [], []
    for i in accepted_idx:
        if not (1 <= i <= len(props)):
            raise SystemExit(f"--accept {i} out of range (1..{len(props)})")
        p = props[i - 1]
        boxes.append({"box": p["box"], "class_id": p["class_id"], "origin": "proposal"})
        origins.append(p["model"])

    for spec in args.add or []:
        parts = [t.strip() for t in spec.split(",")]
        if len(parts) != 5:
            raise SystemExit(f"--add expects 'x1,y1,x2,y2,class_id', got: {spec}")
        x1, y1, x2, y2 = (float(v) for v in parts[:4])
        cid = int(parts[4])
        if cid not in CLASS_NAMES:
            raise SystemExit(f"class id {cid} not in {sorted(CLASS_NAMES)}")
        w, h = entry["size"]
        x1, x2 = sorted((max(0.0, min(x1, w)), max(0.0, min(x2, w))))
        y1, y2 = sorted((max(0.0, min(y1, h)), max(0.0, min(y2, h))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise SystemExit(f"'--add' degenerate box: {spec}")
        boxes.append({"box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                      "class_id": cid, "origin": "manual"})

    GT_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "image_id": args.image_id,
        "path": entry["path"],
        "scene": entry["scene"],
        "split": entry["split"],
        "size": entry["size"],
        "status": "verified",
        "verified_by": args.verified_by,
        "verification_method": args.method,
        "verified_on": _now(),
        "counts": {CLASS_NAMES[c]: sum(1 for b in boxes if b["class_id"] == c) for c in CLASS_NAMES},
        "n_boxes": len(boxes),
        "n_proposals_seen": len(props),
        "n_proposals_rejected": len(reject),
        "boxes": boxes,
        "notes": args.notes or "",
    }
    (GT_DIR / f"{args.image_id}.json").write_text(json.dumps(doc, indent=2))
    print(f"[finalize] {args.image_id}: {len(boxes)} verified boxes "
          f"({len(reject)} proposals rejected, {len(args.add or [])} manual additions) -> "
          f"labels/detection/gt/{args.image_id}.json")
    return 0


# ------------------------------------------------------------------------ stats
def cmd_stats(args) -> int:
    manifest = load_manifest()
    rows = []
    for e in manifest["images"]:
        gt = GT_DIR / f"{e['id']}.json"
        prop = PROP_DIR / f"{e['id']}.json"
        n_gt = len(json.loads(gt.read_text())["boxes"]) if gt.exists() else None
        n_prop = len(json.loads(prop.read_text())["proposals"]) if prop.exists() else None
        rows.append((e["id"], e["split"], e["scene"], n_prop, n_gt))

    print(f"{'image':42s} {'split':5s} {'scene':15s} {'prop':>5s} {'gt':>5s}")
    for r in rows:
        print(f"{r[0]:42s} {r[1]:5s} {r[2]:15s} "
              f"{'-' if r[3] is None else r[3]:>5} {'-' if r[4] is None else r[4]:>5}")

    for split in ("dev", "eval"):
        counts = {name: 0 for name in CLASS_NAMES.values()}
        n_img = 0
        for e in manifest["images"]:
            if e["split"] != split:
                continue
            gt = GT_DIR / f"{e['id']}.json"
            if not gt.exists():
                continue
            n_img += 1
            for b in json.loads(gt.read_text())["boxes"]:
                counts[CLASS_NAMES[b["class_id"]]] += 1
        print(f"\n{split}: {n_img} labelled image(s)")
        for name, n in counts.items():
            print(f"   {name:11s} {n}")

    unverified = [e["id"] for e in manifest["images"]
                  if not (GT_DIR / f"{e['id']}.json").exists()]
    if unverified:
        print(f"\n{len(unverified)} image(s) have no verified labels yet -> excluded from metrics")
    return 0


# ------------------------------------------------------------------- yolo export
def cmd_export_yolo(args) -> int:
    manifest = load_manifest()
    out_dir = LABEL_DIR / "yolo_export" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES[c] for c in sorted(CLASS_NAMES)))
    n = 0
    for e in manifest["images"]:
        if e["split"] != args.split:
            continue
        gt = GT_DIR / f"{e['id']}.json"
        if not gt.exists():
            continue
        doc = json.loads(gt.read_text())
        w, h = doc["size"]
        lines = []
        for b in doc["boxes"]:
            x1, y1, x2, y2 = b["box"]
            lines.append(f"{b['class_id']} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        (out_dir / f"{e['id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    print(f"[export-yolo] {n} label files -> {out_dir.relative_to(ROOT)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Bounding-box labelling workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init-manifest", help="scan local images and write the manifest")
    p.set_defaults(func=lambda a: (init_manifest(), 0)[1])

    p = sub.add_parser("propose", help="generate unverified model proposals")
    p.add_argument("--split", default="eval", choices=["eval", "dev", "all"])
    p.add_argument("--models", nargs="+", default=["yolo11n.pt"])
    p.add_argument("--conf", type=float, default=0.12)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default=config.DEVICE)
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser("render", help="write numbered review images")
    p.add_argument("--image-id", default="all")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("finalize", help="write verified ground truth for one image")
    p.add_argument("--image-id", required=True)
    p.add_argument("--accept", default="all", help="'all', 'none', or 1-based proposal indices")
    p.add_argument("--reject", default="", help="1-based proposal indices to drop")
    p.add_argument("--add", nargs="*", default=[], help="manual boxes: x1,y1,x2,y2,class_id")
    p.add_argument("--method", default="proposal-assisted visual review",
                   help="'manual' when no proposals were used")
    p.add_argument("--verified-by", default="agent-visual")
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_finalize)

    p = sub.add_parser("stats", help="labelling coverage")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export-yolo", help="export verified labels to YOLO txt")
    p.add_argument("--split", default="eval", choices=["eval", "dev"])
    p.set_defaults(func=cmd_export_yolo)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
