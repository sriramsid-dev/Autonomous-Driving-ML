"""
Extract the frames used for lane evaluation from the approved clips.

    python scripts/extract_lane_frames.py

Writes lossless PNGs to test_data/lane_clips/frames/ (git-ignored; the clips
themselves are also git-ignored). Frame selection is recorded here so the exact
set can be rebuilt, and `labels/lane/clips_gt.json` refers to these filenames.

Clip E reports an unreliable frame count through the webm demuxer (it claims
1000 fps), so its frames are selected by timestamp instead of index.
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "test_data" / "lane_clips"
FRAMES = CLIPS / "frames"

# (frame id, clip, selection)
#   ("frame", n) -> seek by frame index
#   ("time", s)  -> seek by timestamp in seconds
PICKS = [
    ("C01", "C_conor_pass.webm", ("frame", 21)),
    ("C02", "C_conor_pass.webm", ("frame", 86)),
    ("C03", "C_conor_pass.webm", ("frame", 151)),
    ("C04", "C_conor_pass.webm", ("frame", 216)),
    ("C05", "C_conor_pass.webm", ("frame", 345)),
    ("D01", "D_ljubljana_kranj.webm", ("frame", 410)),
    ("D02", "D_ljubljana_kranj.webm", ("frame", 718)),
    ("D03", "D_ljubljana_kranj.webm", ("frame", 1026)),
    ("D04", "D_ljubljana_kranj.webm", ("frame", 1333)),
    ("D05", "D_ljubljana_kranj.webm", ("frame", 560)),
    ("D06", "D_ljubljana_kranj.webm", ("frame", 1150)),
    ("C06", "C_conor_pass.webm", ("frame", 110)),
    ("C07", "C_conor_pass.webm", ("frame", 190)),
    ("C08", "C_conor_pass.webm", ("frame", 265)),
    ("C09", "C_conor_pass.webm", ("frame", 310)),
    ("E01", "E_night_motorway.webm", ("time", 20.0)),
    ("E02", "E_night_motorway.webm", ("time", 55.0)),
    ("E03", "E_night_motorway.webm", ("time", 95.0)),
    ("E04", "E_night_motorway.webm", ("time", 130.0)),
    # Round 2 (approved 2026-10-25): F = daytime US highway (straight, curves,
    # low sun / shadow), G = night urban interstate. Both posters were inspected
    # before download; E above stays excluded as an overhead shot.
    ("F01", "F_us33_west_virginia.webm", ("frame", 164)),
    ("F02", "F_us33_west_virginia.webm", ("frame", 982)),
    ("F03", "F_us33_west_virginia.webm", ("frame", 1145)),
    ("F04", "F_us33_west_virginia.webm", ("frame", 1309)),
    ("G01", "G_i235_okc_night.webm", ("frame", 0)),
    ("G02", "G_i235_okc_night.webm", ("frame", 2273)),
    ("G03", "G_i235_okc_night.webm", ("frame", 2557)),
    ("G04", "G_i235_okc_night.webm", ("frame", 2841)),
]


def main() -> int:
    FRAMES.mkdir(parents=True, exist_ok=True)
    caps = {}
    written = []
    for fid, clip, how in PICKS:
        path = CLIPS / clip
        if not path.exists():
            print(f"MISSING {clip} - run scripts/fetch_lane_clips.py")
            return 1
        cap = caps.get(clip)
        if cap is None:
            cap = caps[clip] = cv2.VideoCapture(str(path))
        if how[0] == "frame":
            cap.set(cv2.CAP_PROP_POS_FRAMES, how[1])
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, how[1] * 1000.0)
        ok, frame = cap.read()
        if not ok:
            print(f"FAILED to read {fid} from {clip}")
            continue
        out = FRAMES / f"{fid}.png"
        cv2.imwrite(str(out), frame)
        written.append(fid)
        print(f"{fid}: {frame.shape[1]}x{frame.shape[0]} -> {out.name}")
    for cap in caps.values():
        cap.release()
    print(f"\n{len(written)} frames in {FRAMES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
