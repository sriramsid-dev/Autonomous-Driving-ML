"""
Detector proposals for the v2 held-out traffic-light set.

    python scripts/make_tl_v2_proposals.py [--imgsz 1280]

Runs the module's detector over test_data/tl_v2/*.jpg and writes every box it
proposes to labels/traffic_lights/v2/proposals.json plus a crop per box. These
are *proposals*, not labels: labels/traffic_lights/v2/gt.json is written by hand
after looking at the crops and the source photographs.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.traffic_light_detection import TrafficLightDetector  # noqa: E402

SRC = ROOT / "test_data" / "tl_v2"
OUT = ROOT / "labels" / "traffic_lights" / "v2"


def sheet(crops, cols=8, cell=150):
    tiles = []
    for idx, crop in crops:
        h, w = crop.shape[:2]
        s = min(cell / max(1, w), cell / max(1, h))
        t = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((cell + 18, cell, 3), 30, np.uint8)
        canvas[:t.shape[0], :t.shape[1]] = t
        cv2.putText(canvas, str(idx), (2, cell + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.full((cell + 18, cell, 3), 30, np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.20)
    args = ap.parse_args()

    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    det = TrafficLightDetector(imgsz=args.imgsz, conf_thresh=args.conf)
    images = sorted(SRC.glob("*.jpg"))
    if not images:
        print("no images: run scripts/fetch_tl_v2.py first")
        return 1

    proposals, crops = [], []
    idx = 0
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"unreadable: {path.name}")
            continue
        dets, ms = det.detect(frame)
        print(f"{path.name}  {frame.shape[1]}x{frame.shape[0]}  {len(dets)} boxes  {ms:.0f} ms")
        for d in dets:
            idx += 1
            crop = d["crop"]
            cv2.imwrite(str(OUT / "crops" / f"{idx:02d}.jpg"), crop)
            proposals.append({
                "index": idx, "src": str(path.relative_to(ROOT)).replace("\\", "/"),
                "box": [int(v) for v in d["box"]], "w": crop.shape[1], "h": crop.shape[0],
                "detector_conf": round(d["confidence"], 3), "verified": False,
            })
            crops.append((idx, crop))

    (OUT / "proposals.json").write_text(json.dumps({
        "generated_by": "scripts/make_tl_v2_proposals.py",
        "imgsz": args.imgsz, "conf": args.conf,
        "warning": "detector output only - NOT labels; do not evaluate against this file",
        "proposals": proposals,
    }, indent=2), encoding="utf-8")
    cv2.imwrite(str(OUT / "sheet_all.jpg"), sheet(crops))
    print(f"\n{idx} proposals -> {OUT/'sheet_all.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
