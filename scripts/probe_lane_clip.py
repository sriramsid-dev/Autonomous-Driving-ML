"""Probe a lane clip and build a contact sheet of evenly spaced frames.

    python scripts/probe_lane_clip.py F_us33_west_virginia.webm [G_...webm] [--n 12]

Reports what OpenCV can read (frames, fps, resolution, duration) and writes one
JPEG contact sheet per clip to outputs/runs/lane_v3/, with the frame index
printed on every tile. The sheet exists so a human can see, *before* any
labelling, whether the footage is forward-facing, whether painted lane markings
are visible, and which indices are worth extracting as evaluation frames.
Nothing is labelled here and no model is loaded.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "test_data" / "lane_clips"
OUT = ROOT / "outputs" / "runs" / "lane_v3"


def sheet(name, indices, frames, cols=4):
    tiles = []
    for idx, fr in indices:
        t = cv2.resize(fr, (480, int(480 * fr.shape[0] / fr.shape[1])))
        cv2.rectangle(t, (0, 0), (150, 30), (0, 0, 0), -1)
        cv2.putText(t, f"{name} #{idx}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out = OUT / f"sheet_{name}.jpg"
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 86])
    return out


def probe(path, n):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"{path.name}: {w}x{h}  frames={total}  fps={fps:.2f}  "
          f"duration={total / fps if fps else 0:.1f}s")

    idxs = sorted({max(0, min(total - 1, int(round(i * (total - 1) / (n - 1)))))
                   for i in range(n)}) if total > 1 else [0]
    picked = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            print(f"  FAILED to read frame {idx}")
            continue
        picked.append((idx, fr))
    cap.release()
    print(f"  read {len(picked)} frames: {[i for i, _ in picked]}")
    if picked:
        print(f"  wrote {sheet(path.stem.split('_')[0], picked, n and len(picked))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for c in args.clips:
        p = CLIPS / c
        if not p.exists():
            print(f"MISSING {c} - run scripts/fetch_lane_clips.py")
            continue
        probe(p, args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
