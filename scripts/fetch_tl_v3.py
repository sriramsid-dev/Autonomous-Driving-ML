"""
Fetch the small, free-licensed traffic-light set used for the v3 evaluation.

    python scripts/fetch_tl_v3.py

Downloads 12 Wikimedia Commons photographs at 1600 px width (~6 MB total) into
test_data/tl_v3/. Titles, authors, licences and page URLs are written to
test_data/tl_v3/sources.json so attribution travels with the data.

These 12 are drawn from the same Commons candidate pool as v2 but none of them
appear in v2 or in the dev crops, so the classifier had never seen them when
its thresholds were fixed. The API is not queried at all: the pool already
holds the thumb URL and licence for every title, which avoids the Commons
search rate limit.
"""

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_data" / "tl_v3"
POOL = ROOT / "outputs" / "runs" / "tl_v2" / "commons_candidates.json"

# Indices into the candidate pool. Chosen for variety, not for difficulty:
# straight-on urban heads, distant heads in wide shots, night scenes, an amber
# signal, red signals, a green sign, and cluttered roadside clutter.
PICKS = [
    ("Darwin (AU), Knuckey St-Mitchell St, Traffic Light -- 2019 -- 3640-1.jpg", 1),
    ("Traffic Light, Vienna.jpg", 4),
    ("Traffic lights at Manchester St, Christchurch, New Zealand 04.jpg", 8),
    ("Yellow traffic signal (9722336142).jpg", 9),
    ("Rebiechowo traffic lights by night - panoramio.jpg", 14),
    ("Traffic Lights, NYC (6239701931).jpg", 16),
    ("Traffic light red and yellow Drammen (2).jpg", 35),
    ("Northernmost traffic light in North America, Inuvik, NT.jpg", 40),
    ("Traffic light Banyakan Ngawi.jpg", 43),
    ("Traffic light at US319 and East Ivan from East Ivan under construction.jpg", 44),
    ("Traffic light with a green sign (Kiev).jpg", 47),
    ("Traffic at Night in Valparaiso, IN.JPG", 17),
]


def main() -> int:
    if not POOL.exists():
        print(f"missing candidate pool: {POOL}")
        return 1
    pool = {c["title"][5:]: c for c in json.loads(POOL.read_text())}
    OUT.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "perception-eval-script/1.0 (local research)"})
    sources = []
    for i, (name, _idx) in enumerate(PICKS, 1):
        cand = pool.get(name)
        if cand is None:
            print(f"SKIP {name}: not in the candidate pool")
            continue
        slug = f"{i:02d}_" + "".join(c if c.isalnum() else "_" for c in name)[:40] + ".jpg"
        path = OUT / slug
        if not path.exists():
            r = session.get(cand["thumb"], timeout=120)
            if r.status_code != 200 or len(r.content) < 20_000:
                print(f"SKIP {name}: HTTP {r.status_code}, {len(r.content)} bytes")
                time.sleep(2.0)
                continue
            path.write_bytes(r.content)
            time.sleep(1.5)  # be a good citizen
        sources.append({
            "file": slug,
            "title": name,
            "author": cand.get("artist") or "unknown",
            "license": cand.get("license") or "see page",
            "page": cand.get("page") or "",
            "bytes": path.stat().st_size,
            "note": "crop verification: labels in labels/traffic_lights/v3/gt.json",
        })
        print(f"ok   {slug}  {path.stat().st_size // 1024} KB  {cand.get('license')}")

    (OUT / "sources.json").write_text(json.dumps(sources, indent=1))
    total = sum(s["bytes"] for s in sources)
    print(f"\n{len(sources)} photographs, {total / 1e6:.1f} MB, attribution in {OUT / 'sources.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
