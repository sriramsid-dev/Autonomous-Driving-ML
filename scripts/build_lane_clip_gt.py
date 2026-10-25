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
DRAFT = ROOT / "labels" / "lane" / "clips_draft.json"     # round 1: clips C and D
DRAFT2 = ROOT / "labels" / "lane" / "clips_draft2.json"   # round 2: clips F and G
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

# Round 2 (clips F and G). F is a US two-lane road: the double yellow centre
# line is ego's LEFT boundary and the white edge line its RIGHT. G is a night
# interstate; only the solid right edge line was traceable - the lane divider
# is dashed and the road band is only ~150 rows tall, so its traces did not
# survive the overlay check.
VERIFIED2 = {
    "F01": {"left": "left", "right": "right"},
    "F02": {"left": "left"},
    "F03": {"left": "left"},
    "G01": {"right": "right"},
    "G02": {"right": "right"},
}

SCENE = {
    "C": ("wet single-track mountain road, ego keeps the yellow centre line on its left",
          "day, overcast, wet surface", "left-hand curve then right-hand curve"),
    "D": ("motorway, ego in the right-hand lane",
          "dusk, low sun behind, dark asphalt", "straight with a slight right-hand bend"),
    "F": ("two-lane rural US highway, ego between the double yellow centre line and the white edge line",
          "late afternoon, low sun, long shadows across the road",
          "F01 straight, F02 left-hand curve, F03 right-hand curve"),
    "G": ("multi-lane urban interstate at night, ego between two white lines",
          "night, street lighting plus headlight illumination",
          "G01 straight, G02 straight under an overhead sign gantry"),
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
    "F02": ["right"],   # white edge line: trace landed on the guardrail
    "F03": ["right"],   # ditto
    "G01": ["left"],    # dashed divider: too sparse to verify at this resolution
    "G02": ["left"],
}


CLIP_OF = {"C": "C_conor_pass", "D": "D_ljubljana_kranj",
           "F": "F_us33_west_virginia", "G": "G_i235_okc_night"}


def main() -> int:
    if not DRAFT.exists():
        print(f"missing {DRAFT} - run scripts/label_lane_marks.py first")
        return 1
    drafts = {**json.loads(DRAFT.read_text(encoding="utf-8"))}
    if DRAFT2.exists():
        drafts.update(json.loads(DRAFT2.read_text(encoding="utf-8")))
    verified = {**VERIFIED, **VERIFIED2}

    images = []
    for fid in sorted(verified):
        if fid not in drafts:
            print(f"SKIP {fid}: not in the draft")
            continue
        entry = drafts[fid]
        boundaries = []
        for draft_side, pub_side in verified[fid].items():
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
            "clip": CLIP_OF[fid[0]],
            "scene": scene,
            "condition": condition,
            "layout": layout,
            "boundaries": boundaries,
            "unavailable": UNAVAILABLE.get(fid, []),
            "verified_by": "overlay of the traced points on the frame "
                           "(outputs/runs/lane_v2/overlays/ for C and D, "
                           "outputs/runs/lane_v3/verify*.jpg for F and G)",
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
            "clips": ("test_data/lane_clips: C = Conor Pass (CC BY-SA 4.0), D = Ljubljana to Kranj "
                      "(CC BY 3.0), F = US 33 West Virginia (CC BY 3.0, Andy Arthur), "
                      "G = I-235 North OKC (CC BY-SA 4.0, TheDoctorWho). Sources, licences and "
                      "attribution: test_data/lane_clips/sources.json"),
            "why_few_frames": ("Round 1 traced 13 C/D frames, of which 8 survived the overlay check. "
                               "Round 2 extracted 8 more from clips F and G and traced 12 boundaries; "
                               "7 survived. Discarded rather than guessed: traces that followed a "
                               "guardrail (F02/F03 right edge line), the road shadow, a road sign or a "
                               "lamp post, and C01 whose trace covered only 12 rows. F04 was excluded "
                               "because the low sun leaves the centre line in deep shadow, and G03 "
                               "because a gore/merge area makes the ego lane boundary ambiguous."),
            "limitations": ("13 frames from 4 clips: a wet daylight mountain road, a dusk motorway, "
                            "a late-afternoon US two-lane highway and a night urban interstate. No rain, "
                            "no construction zone, no snow. The C clip and the G clip contribute one "
                            "boundary per frame, so ego-pair checks there can only ever be partial and "
                            "are reported as coverage rather than as pair detection."),
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
