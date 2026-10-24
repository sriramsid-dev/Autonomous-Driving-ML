"""
Fetch the small, free-licensed traffic-light set used for the v2 held-out test.

    python scripts/fetch_tl_v2.py

Downloads 12 Wikimedia Commons photographs at 1600 px width (~5 MB total) into
test_data/tl_v2/, and records title, author, licence and page URL in
test_data/tl_v2/sources.json so attribution travels with the data.

Nothing is downloaded unless you run this. The originals are git-ignored; the
crops that the detector produces from them are committed together with their
labels and this attribution.
"""
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_data" / "tl_v2"
PICKS = [
    "Flashing yellow traffic light 3086.jpg",
    "Amber traffic signal, Stamford Road, Singapore - 20111210-01.jpg",
    "Traffic light yellow Drammen.jpg",
    "LED traffic light on red.jpg",
    "UK Red Traffic Light Signal - geograph.org.uk - 5099256.jpg",
    "Traffic lights at night.jpg",
    "Traffic lights at night1.jpg",
    "Traffic light at Jeffrey Road and Smoketree.jpg",
    "Cyclist traffic light at the oval.jpg",
    "Traffic light in Kumasi.jpg",
    "Traffic light (Frankfurt Wolkenkratzerfestival).jpg",
    "Out-of-focus dashboard and traffic lights at night 01.jpg",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = {c["title"][5:]: c for c in json.loads(
        (ROOT / "outputs" / "runs" / "tl_v2" / "commons_candidates.json").read_text())}
    session = requests.Session()
    session.headers.update({"User-Agent": "perception-eval-script/1.0 (local research)"})
    sources = []
    for i, name in enumerate(PICKS, 1):
        cand = cands.get(name)
        if cand is None:
            print(f"SKIP {name}: not in candidate list (re-run the Commons query)")
            continue
        slug = f"{i:02d}_" + "".join(c if c.isalnum() else "_" for c in name)[:40] + ".jpg"
        path = OUT / slug
        if not path.exists():
            r = session.get(cand["thumb"], timeout=120)
            r.raise_for_status()
            if len(r.content) < 20_000:
                print(f"SKIP {name}: suspiciously small response")
                continue
            path.write_bytes(r.content)
        sources.append({
            "file": slug,
            "title": name,
            "author": cand["artist"],
            "license": cand["license"],
            "page": cand["page"],
            "bytes": path.stat().st_size,
            "note": "1600 px rendering of the Commons file; crop is an adaptation",
        })
        print(f"{slug}  {path.stat().st_size // 1024} KB  [{cand['license']}]")
    (OUT / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"\n{len(sources)} images, {sum(s['bytes'] for s in sources)//1024} KB total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
