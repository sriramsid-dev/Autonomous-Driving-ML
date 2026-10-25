"""
Screen small, freely licensed driving videos for lane-marking suitability.

    python scripts/find_lane_clips.py

Queries Wikimedia Commons for video files about driving, then downloads only
each file's **poster thumbnail** (a few tens of KB) to test_data/lane_clips/
posters/. A video is not downloaded here: the poster frame is enough to reject
clips that are drone shots, in-car vlogs, or unmarked rural roads, so the large
download only happens for a clip that already looks usable.

Writes test_data/lane_clips/candidates.json with title, page, licence, author,
byte size, duration/resolution where stated, and the poster path.
"""

import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_data" / "lane_clips"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
POSTERS = OUT / "posters"
API = "https://commons.wikimedia.org/w/api.php"

QUERIES = [
    "filetype:video dashcam",
    "filetype:video driving motorway",
    "filetype:video car driving road",
    "filetype:video autobahn fahrt",
    "filetype:video driving lane markings",
    "filetype:video windshield road",
]
MAX_MB = 200


def api(session, params, tries=4):
    for attempt in range(tries):
        r = session.get(API, params={"format": "json", **params}, timeout=90)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        print(f"   retry status={r.status_code}")
        time.sleep(3 + 4 * attempt)
    return {}


def clean(html):
    return re.sub("<[^>]+>", "", html or "").strip()[:70]


def main() -> int:
    POSTERS.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "perception-eval-script/1.0 (local research)"})

    # --titles "File:A.webm" "File:B.webm" screens named files only (no search),
    # so a shortlist from any source can be poster-checked without downloading it.
    named = len(sys.argv) > 2 and sys.argv[1] == "--titles"
    if named:
        titles = list(dict.fromkeys(sys.argv[2:]))
        print(f"screening {len(titles)} named file(s)")
    else:
        titles = []
        for q in QUERIES:
            data = api(session, {"action": "query", "list": "search", "srsearch": q,
                                 "srnamespace": "6", "srlimit": "20"})
            hits = [h["title"] for h in data.get("query", {}).get("search", [])]
            print(f"{q!r}: {len(hits)} hits")
            titles += hits
            time.sleep(2.0)
    titles = list(dict.fromkeys(titles))

    cands = []
    for i in range(0, len(titles), 25):
        data = api(session, {"action": "query", "titles": "|".join(titles[i:i + 25]),
                             "prop": "imageinfo",
                             "iiprop": "url|size|mime|extmetadata",
                             "iiurlwidth": "480"})
        for pg in (data.get("query", {}).get("pages") or {}).values():
            ii = (pg.get("imageinfo") or [{}])[0]
            if not str(ii.get("mime", "")).startswith("video/"):
                continue
            size = ii.get("size", 0)
            if not size or size > MAX_MB * 1e6:
                continue
            em = ii.get("extmetadata", {})
            cands.append({
                "title": pg["title"][5:],
                "page": ii.get("descriptionurl", ""),
                "url": ii.get("url", ""),
                "poster": ii.get("thumburl", ""),
                "license": em.get("LicenseShortName", {}).get("value", "?"),
                "author": clean(em.get("Artist", {}).get("value", "?")),
                "mb": round(size / 1e6, 1),
                "w": ii.get("width"), "h": ii.get("height"),
                "duration_s": round(float(em.get("Length", {}).get("value", 0) or 0), 1),
            })
        time.sleep(2.0)

    print(f"\n{len(cands)} video candidates under {MAX_MB} MB")
    kept = []
    for c in cands:
        if not c["poster"]:
            continue
        slug = re.sub(r"[^0-9A-Za-z]+", "_", c["title"])[:48] + ".jpg"
        path = POSTERS / slug
        if not path.exists():
            r = session.get(c["poster"], timeout=60)
            if r.status_code != 200 or len(r.content) < 3000:
                continue
            path.write_bytes(r.content)
            time.sleep(0.7)
        c["poster_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        kept.append(c)
        print(f"  {c['mb']:6.1f} MB  {c['w']}x{c['h']}  {c['license'][:12]:12s}  "
              f"{c['title'][:52].encode('ascii','replace').decode()}")

    (OUT / "candidates.json").write_text(json.dumps(kept, indent=1))
    print(f"\n{len(kept)} with posters -> {OUT / 'candidates.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
