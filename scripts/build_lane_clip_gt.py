"""
Build labels/lane/clips_gt.json from the verified paint traces.

    python scripts/label_lane_marks.py        # draft + overlays
    python scripts/build_lane_clip_gt.py      # keep only what passed the overlay check

Only frames whose overlay was inspected and found to sit on the paint are
published here. Everything else - the traces that wandered onto a guardrail, a
road sign, a lamp post or the chevron hatching, and crop C01 whose trace covered
only 12 rows - is dropped rather than guessed at.

Side assignment, and why it is not a coin flip:

  * the camera sits at the horizontal centre of the frame, so the ego lane must
    contain centre x;
  * in the C frames the dashed yellow line is at x ~ 620, i.e. well left of
    centre, while the road extends to the right of it, so that line is the ego
    lane's LEFT boundary. The opposite reading would put the camera outside its
    own lane;
  * in D02 the dashed divider is at x ~ 465 and the solid edge line at x ~ 1700,
    bracketing centre, so those are the left and right boundaries exactly as
    labelled.

A boundary that is not painted, or is occluded at the rows in question, is
simply absent: there is no invented point anywhere in this file.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "labels" / "lane" / "clips_draft.json"
OUT = ROOT / "labels" / "lane" / "clips_gt.json"

# frame -> {draft side: published side}
VERIFIED = {
    "C03": {"right": "left"},
    "C04": {"right": "left"},
    "C05": {"right": "left"},
    "C06": {"right": "left"},
    "C07": {"right": "left"},
    "C08": {"right": "left"},
    "C09": {"right": "left"},
    "D02": {"left": "left", "right": "right"},
}

SCENE = {
    "C": ("wet single-track mountain road, ego keeps the yellow centre line on its left",
          "day, overcast, wet surface", "left-hand curve then right-hand curve"),
    "D": ("motorway, ego in the right-hand lane",
          "dusk, low sun behind, dark asphalt", "straight with a slight right-hand bend"),
}

# boundaries that exist in the frames but were NOT verified, so are unavailable
UNAVAILABLE = {
    "C03": ["right"],
    "C04": ["right"],
    "C05": ["right"],
    "C06": ["right"],
    "C07": ["right"],
    "C08": ["right"],
    "C09": ["right"],
    "D02": [],
}


def main() -> int:
    if not DRAFT.exists():
        print(f"missing {DRAFT} - run scripts/label_lane_marks.py first")
        return 1
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))

    images = []
    for fid in sorted(VERIFIED):
        if fid not in draft:
            print(f"SKIP {fid}: not in the draft")
            continue
        entry = draft[fid]
        boundaries = []
        for draft_side, pub_side in VERIFIED[fid].items():
            b = entry["boundaries"].get(draft_side)
            if not b or len(b["rows"]) < 4:
                continue
            boundaries.append({
                "side": pub_side,
                "kind": b["kind"],
                "rows": sorted(b["rows"], key=lambda r: r["y"]),
            })
        if not boundaries:
            continue
        scene, condition, layout = SCENE[fid[0]]
        images.append({
            "path": entry["frame"],
            "size": entry["size"],
            "clip": "C_conor_pass" if fid.startswith("C") else "D_ljubljana_kranj",
            "scene": scene,
            "condition": condition,
            "layout": layout,
            "boundaries": boundaries,
            "unavailable": UNAVAILABLE.get(fid, []),
            "verified_by": "overlay of the traced points on the frame (outputs/runs/lane_v2/overlays/)",
        })

    doc = {
        "provenance": {
            "labelled_by": "coding agent: seeded paint trace (scripts/label_lane_marks.py) plus visual check of the drawn overlay",
            "labelled_on": "2026-10-25",
            "method": ("a seed row is scanned for paint runs of the right colour, the trace walks up and down "
                       "from there, a quadratic is fitted and every surviving point is drawn back onto the frame. "
                       "The lane model is never loaded by the labeller."),
            "coordinate_system": "original frame pixels (x, y), origin top-left; x is the centre of the paint run",
            "uncertainty_px": 5,
            "clips": "test_data/lane_clips: C = Conor Pass (CC BY-SA 4.0), D = Ljubljana to Kranj (CC BY 3.0)",
            "why_few_frames": ("10 frames were extracted and traced, 13 in total including four extra C frames; "
                               "only the 8 below survived the overlay check. Traces that followed a guardrail, a "
                               "road sign, a lamp post or the chevron hatching were discarded, and frame C01 was "
                               "dropped because its trace covered only 12 rows."),
            "limitations": ("8 frames from 2 clips, one wet daylight mountain road and one dusk motorway. "
                            "No night, no rain, no construction. The C clip contributes one boundary per frame, "
                            "so ego-pair checks there must fail by construction and are reported separately."),
        },
        "images": images,
    }
    OUT.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    n_pts = sum(len(b["rows"]) for i in images for b in i["boundaries"])
    print(f"{len(images)} frames, "
          f"{sum(len(i['boundaries']) for i in images)} boundaries, {n_pts} points -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
