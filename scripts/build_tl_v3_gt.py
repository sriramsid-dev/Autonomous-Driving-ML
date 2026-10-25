"""
Materialise the v3 evaluation set: crops plus the hand-written labels.

    python scripts/build_tl_v3_gt.py

The labels below were assigned by eye, one crop at a time, from the contact
sheet and the zoomed views produced by `make_tl_v3_proposals.py`. The detector
only proposed *where to look*; it never supplied a label. Where a crop could not
be resolved by eye (a 16x45 px head, a pair of lamps lit together, a rainy
blur) the label is `ambiguous`, which the evaluator scores separately and never
as a correct answer.

Vocabulary matches the other sets so `scripts/eval_traffic_lights.py` reads it:

    red / yellow / green      a vehicle signal whose emitting lamp is verifiable
    ambiguous                 lamps of two states lit, or state not resolvable
    unlit                     a real signal, no lamp visibly emitting
    not_a_traffic_light       back of a head, a backplate, a basket, a pole
"""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SET_DIR = ROOT / "labels" / "traffic_lights" / "v3"
SRC = ROOT / "test_data" / "tl_v3"

# key: "<first two characters of image stem>.<detector index>" as printed on the sheet
LABELS = {
    # --- vehicle signals whose emitting lamp is verifiable -------------------
    "02.0": ("green", "pedestrian go figure lit; upper lens unlit"),
    "06.1": ("green", "NYC head cropped to the backplate; green emitting at the lower edge"),
    "12.0": ("green", "night, rain on the lens; green lamp lit"),
    "12.2": ("green", "night, rain on the lens; green lamp lit"),
    "09.0": ("red", "top red lamp lit with a white core; lower lens dark"),
    "11.0": ("red", "top red lamp lit; middle and lower lenses dark"),
    "03.0": ("yellow", "bicycle signal; amber glyph lit, upper and lower lenses dark"),
    "03.4": ("yellow", "bicycle signal; amber glyph lit"),
    "04.3": ("yellow", "lower amber lens lit and textured; upper lens dark red, unlit"),
    # --- genuinely ambiguous -------------------------------------------------
    "03.2": ("ambiguous", "red pedestrian figure lit AND amber countdown lit"),
    "07.0": ("ambiguous", "red lamp and amber lamp both clearly emitting"),
    "08.0": ("ambiguous", "16x45 px head, 4 aspects, every band bright - state not verifiable"),
    "12.1": ("ambiguous", "night, rain, defocused amber-orange blob - state not verifiable"),
    # --- real signal, nothing emitting --------------------------------------
    "03.1": ("unlit", "housing edge, no lens"),
    "03.3": ("unlit", "dark housing, no lens visible"),
    "03.5": ("unlit", "dark housing, lenses unlit"),
    "03.6": ("unlit", "dark housing, lenses unlit"),
    "04.0": ("unlit", "three-lamp head, all lenses unlit"),
    "04.1": ("unlit", "head seen from above, all lenses unlit"),
    "04.2": ("unlit", "head seen from the side, all lenses unlit"),
    "08.2": ("unlit", "yellow head, lenses dark"),
    "11.1": ("unlit", "dark housing, lens unlit"),
    # --- not a signal at all -------------------------------------------------
    "02.1": ("not_a_traffic_light", "blank pole/mount"),
    "02.2": ("not_a_traffic_light", "pedestrian push-button box"),
    "03.7": ("not_a_traffic_light", "pole with a white strip"),
    "06.0": ("not_a_traffic_light", "NYC yellow backplate edge"),
    "06.2": ("not_a_traffic_light", "NYC yellow backplate"),
    "08.1": ("not_a_traffic_light", "yellow pole/backplate"),
    "08.3": ("not_a_traffic_light", "back of a yellow head with a bracket"),
    "08.4": ("not_a_traffic_light", "hanging basket beside a yellow plate"),
    "09.1": ("not_a_traffic_light", "pole with a specular speck"),
    "10.0": ("not_a_traffic_light", "back of a head against the sky"),
    "10.1": ("not_a_traffic_light", "back of a horizontal head"),
    "10.2": ("not_a_traffic_light", "back of a head"),
    "10.3": ("not_a_traffic_light", "back/side of a head"),
    "10.4": ("not_a_traffic_light", "back of a head"),
    "11.2": ("not_a_traffic_light", "back of a housing"),
    "11.3": ("not_a_traffic_light", "back of a housing"),
}


def main() -> int:
    proposals = json.loads((SET_DIR / "proposals.json").read_text())
    crops_dir = SET_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for p in proposals:
        stem = Path(p["image"]).stem
        key = f"{stem[:2]}.{p['index']}"
        if key not in LABELS:
            print(f"UNLABELLED {key} ({p['image']}) - skipped")
            continue
        label, note = LABELS[key]
        img = cv2.imread(str(SRC / p["image"]))
        if img is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        name = f"{key.replace('.', '_')}.jpg"
        cv2.imwrite(str(crops_dir / name), crop)
        samples.append({
            "index": int(key.split(".")[1]),
            "key": key,
            "image": p["image"],
            "crop": str((crops_dir / name).relative_to(ROOT)).replace("\\", "/"),
            "box": p["box"],
            "det_conf": p["det_conf"],
            "label": label,
            "note": note,
        })

    doc = {
        "set": "v3",
        "description": (
            "11 Wikimedia Commons photographs (CC BY / CC BY-SA / CC0 / public domain, "
            "attribution in test_data/tl_v3/sources.json), 38 detector-proposed regions, "
            "labels assigned by hand from the contact sheet and zoomed crops. No threshold "
            "in config.py was chosen using this set: it was materialised after v1.5 was "
            "frozen and scored in a single run."
        ),
        "labelling": "manual, by eye; detector proposals only located the regions",
        "counts": {},
        "samples": samples,
    }
    for s in samples:
        doc["counts"][s["label"]] = doc["counts"].get(s["label"], 0) + 1
    (SET_DIR / "gt.json").write_text(json.dumps(doc, indent=1))
    print(f"{len(samples)} labelled crops -> {SET_DIR / 'gt.json'}")
    print("counts:", doc["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
