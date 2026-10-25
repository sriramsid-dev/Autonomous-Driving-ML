"""
Fetch the two small, freely licensed driving clips used for lane evaluation.

    python scripts/fetch_lane_clips.py          # download (~114 MB)
    python scripts/fetch_lane_clips.py --check  # report licence/size, download nothing

Both are Wikimedia Commons files with direct download routes and stated
licences. Originals are git-ignored (test_data/lane_clips/*.webm); the
recorded metadata in sources.json is committed so attribution travels with any
frames extracted from them.

Clip C (Conor Pass, Dingle Peninsula)   2.5 MB, 1920x1080, CC BY-SA 4.0
Clip D (Driving from Ljubljana to Kranj) 62.6 MB, 1920x1080, CC BY 3.0, dusk motorway
Clip E (Cars driving at night)         113.6 MB, 1920x1080, CC BY 3.0, night motorway

The clips were screened first with `scripts/find_lane_clips.py`, which fetches
only each candidate's poster frame. No other dataset is downloaded and nothing
here is trained.
"""

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_data" / "lane_clips"

# Round 1 (clips A and B) was downloaded, inspected frame by frame and rejected:
# A is a winter drive with no painted lane line anywhere (48 frames sampled
# across all 209 s) and B is aerial drone footage with no ego lane. See
# EVALUATION.md 13. Their URLs are in the git history of this file.
#
# The clips below were chosen from scripts/find_lane_clips.py by inspecting each
# candidate's poster frame first, so the video download only happens for clips
# whose first frame already shows painted markings from a forward-facing car.
CLIPS = [
    {
        "id": "C",
        "file": "C_conor_pass.webm",
        "commons_title": "File:Conor Pass-1012864, Dingle Peninsula, Co. Kerry, Ireland.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Conor_Pass-1012864,_Dingle_Peninsula,_Co._Kerry,_Ireland.webm",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Conor_Pass-1012864,_Dingle_Peninsula,_Co._Kerry,_Ireland.webm",
        "license": "CC BY-SA 4.0",
        "author": "see Commons file page",
        "expected_bytes": 2_500_000,
        "note": "wet coastal mountain road, painted centre and edge lines, 1920x1080",
    },
    {
        "id": "D",
        "file": "D_ljubljana_kranj.webm",
        "commons_title": "File:Driving from Ljubljana to Kranj.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Driving_from_Ljubljana_to_Kranj.webm",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Driving_from_Ljubljana_to_Kranj.webm",
        "license": "CC BY 3.0",
        "author": "see Commons file page",
        "expected_bytes": 62_600_000,
        "note": "motorway at dusk, dashed lane lines, 1920x1080",
    },
    {
        "id": "E",
        "file": "E_night_motorway.webm",
        "commons_title": "File:Cars driving at night.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Cars_driving_at_night.webm",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Cars_driving_at_night.webm",
        "license": "CC BY 3.0",
        "author": "see Commons file page",
        "expected_bytes": 113_600_000,
        "note": "night motorway in rain, markings lit by headlights, 1920x1080",
    },
]


def main() -> int:
    check_only = "--check" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "perception-eval-script/1.0 (local research)"})

    recorded = []
    for clip in CLIPS:
        path = OUT / clip["file"]
        if not check_only and not path.exists():
            r = session.get(clip["url"], timeout=300, allow_redirects=True)
            if r.status_code != 200 or len(r.content) < 1_000_000:
                print(f"FAIL {clip['id']}: HTTP {r.status_code}, {len(r.content)} bytes")
                continue
            path.write_bytes(r.content)
            print(f"downloaded {clip['file']}: {len(r.content) / 1e6:.1f} MB")
        size = path.stat().st_size if path.exists() else None
        print(f"{clip['id']}  {clip['license']:12s}  "
              f"{'%.1f MB on disk' % (size / 1e6) if size else 'not downloaded'}  {clip['page']}")
        recorded.append({**clip, "bytes": size})

    if not check_only:
        (OUT / "sources.json").write_text(json.dumps(recorded, indent=1))
        total = sum(c["bytes"] or 0 for c in recorded)
        print(f"\n{len(recorded)} clips, {total / 1e6:.1f} MB, metadata in {OUT / 'sources.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
