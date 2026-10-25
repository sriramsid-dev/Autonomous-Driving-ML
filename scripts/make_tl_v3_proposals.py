"""
Propose candidate traffic-light regions in the v3 photographs for manual review.

    python scripts/make_tl_v3_proposals.py

The detector is used only to *find where to look*: it writes one annotated
overview per photograph and a numbered contact sheet of every proposed crop.
Nothing here is ground truth. Labels are written by hand into
labels/traffic_lights/v3/gt.json after inspecting these sheets.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from modules.traffic_light_detection import TrafficLightDetector

SRC = ROOT / "test_data" / "tl_v3"
OUT = ROOT / "labels" / "traffic_lights" / "v3"
SHEET = OUT / "sheet_all.jpg"
TILE = 132


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = TrafficLightDetector(imgsz=max(config.TL_DETECTION_IMGSZ, 1280))
    tiles, meta = [], []
    for img_path in sorted(SRC.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets, _ = det.detect(img)
        view = img.copy()
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(view, str(i), (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            meta.append({"image": img_path.name, "index": i, "box": d["box"],
                         "det_conf": round(d["confidence"], 3)})
        h, w = view.shape[:2]
        s = min(1.0, 1100 / max(h, w))
        cv2.imwrite(str(OUT / f"{img_path.stem}_boxes.jpg"),
                    cv2.resize(view, (int(w * s), int(h * s))))
        print(f"{img_path.name}: {len(dets)} boxes")
        for m in [m for m in meta if m["image"] == img_path.name]:
            x1, y1, x2, y2 = [int(v) for v in m["box"]]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            sc = min(TILE / max(ch, cw), 6.0)
            tile = cv2.resize(crop, (max(1, int(cw * sc)), max(1, int(ch * sc))),
                              interpolation=cv2.INTER_CUBIC)
            canvas = np.full((TILE, TILE, 3), 24, np.uint8)
            oy, ox = (TILE - tile.shape[0]) // 2, (TILE - tile.shape[1]) // 2
            canvas[oy:oy + tile.shape[0], ox:ox + tile.shape[1]] = tile
            cv2.putText(canvas, f"{img_path.stem[:2]}.{m['index']}", (3, TILE - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(canvas)
    if tiles:
        cols = 8
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.full((rows * TILE, cols * TILE, 3), 24, np.uint8)
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t
        cv2.imwrite(str(SHEET), sheet)
    (OUT / "proposals.json").write_text(json.dumps(meta, indent=1))
    print(f"\n{len(meta)} proposals -> {SHEET}")


if __name__ == "__main__":
    main()
