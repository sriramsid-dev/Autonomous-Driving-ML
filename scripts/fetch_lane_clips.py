"""
Fetch the two small, freely licensed driving clips used for lane evaluation.

    python scripts/fetch_lane_clips.py          # download (~114 MB)
    python scripts/fetch_lane_clips.py --check  # report licence/size, download nothing

Both are Wikimedia Commons files with direct download routes and stated
licences. Originals are git-ignored (test_data/lane_clips/*.webm); the
recorded metadata in sources.json is committed so attribution travels with any
frames extracted from them.

Clip A (Dashcam film.webm)     76 MB, 3:29, 720x480,  CC BY-SA 3.0, author Frukko
Clip B (Iran, drive along the Persian Gulf.webm)   38 MB, 0:59, 1280x720, CC BY 3.0

No other dataset is downloaded, and nothing here is trained.
"""

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_data" / "lane_clips"

CLIPS = [
    {
        "id": "A",
        "file": "A_dashcam_film.webm",
        "commons_title": "File:Dashcam film.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Dashcam_film.webm",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Dashcam_film.webm",
        "license": "CC BY-SA 3.0",
        "author": "Frukko",
        "expected_bytes": 76_440_000,
        "note": "dashcam test drive, day and night, 720x480",
    },
    {
        "id": "B",
        "file": "B_persian_gulf_drive.webm",
        "commons_title": "File:Iran, drive along the persian gulf.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Iran,_drive_along_the_persian_gulf.webm",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Iran%2C%20drive%20along%20the%20persian%20gulf.webm",
        "license": "CC BY 3.0",
        "author": "see Commons file page",
        "expected_bytes": 38_310_000,
        "note": "drive along the Persian Gulf coast, 1280x720",
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
