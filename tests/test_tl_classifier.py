"""
Unit tests for the traffic-light state classifier (v1.5).

    python tests/test_tl_classifier.py

Two kinds of case:

  * synthetic renders - drawn here so each failure mode can be isolated and the
    expected answer is unambiguous. They test *code behaviour*, and they make no
    claim about real-world accuracy.
  * real crops from the two labelled sets - only as a regression fence around
    the bug this version fixed (cream housing winning over a lit green lamp).

Exit code is non-zero if any check fails.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.traffic_light_classification import TrafficLightClassifier  # noqa: E402

RESULTS = []

CREAM = (150, 236, 243)     # BGR: warm off-white paint
DARK = (28, 30, 32)
RED = (40, 40, 235)
AMBER = (40, 150, 255)
GREEN = (120, 235, 40)
GLOW = (255, 255, 255)


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")

    return deco


# --------------------------------------------------------------------- render
def housing(h=300, w=130, body=CREAM, lit=(1,), lamp=GREEN, core=True,
            bg=(90, 95, 100), glare=0.0, noise=4.0, dark_lens=True):
    """Draw a 3-lamp signal head. `lit` holds the indices of the lit lamps."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bg
    if glare:
        veil = np.full_like(img, int(255 * glare))
        img = cv2.addWeighted(img, 1.0 - glare, veil, 1.0, 0)
    x0, x1 = int(w * 0.08), int(w * 0.92)
    cv2.rectangle(img, (x0, int(h * 0.03)), (x1, int(h * 0.97)), body, -1)
    r = int(min(w, h) * 0.17)
    centres = [(w // 2, int(h * 0.20)), (w // 2, int(h * 0.50)), (w // 2, int(h * 0.80))]
    for i, (cx, cy) in enumerate(centres):
        cv2.circle(img, (cx, cy), r, DARK if dark_lens else body, -1)
        if i in lit:
            cv2.circle(img, (cx, cy), int(r * 0.86), lamp, -1)
            if core:
                cv2.circle(img, (cx, cy), int(r * 0.42), GLOW, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.6)
    if noise:
        img = np.clip(img.astype(np.float32) + np.random.default_rng(0).normal(0, noise, img.shape), 0, 255)
        img = img.astype(np.uint8)
    return img


def glow_variant(h=300, w=130, body=CREAM, lamp=GREEN, bg=(90, 95, 100)):
    """Lit lamp with a real halo: bright core, saturated falloff, dark bezel."""
    img = housing(h, w, body=body, lit=(), bg=bg)
    r = int(min(w, h) * 0.17)
    cx, cy = w // 2, int(h * 0.80)
    halo = np.zeros_like(img)
    cv2.circle(halo, (cx, cy), int(r * 1.5), lamp, -1)
    halo = cv2.GaussianBlur(halo, (0, 0), r * 0.45)
    img = cv2.addWeighted(img, 0.55, halo, 0.75, 0)
    cv2.circle(img, (cx, cy), int(r * 0.8), lamp, -1)
    cv2.circle(img, (cx, cy), int(r * 0.38), GLOW, -1)
    return cv2.GaussianBlur(img, (0, 0), 1.4)


# ------------------------------------------------------------------- synthetic
@check("invalid and degenerate input returns unknown")
def test_invalid():
    clf = TrafficLightClassifier()
    for bad in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((1, 1, 3), np.uint8),
                np.zeros((4, 4, 3), np.uint8)):
        state, conf = clf.classify(bad)
        assert state == "unknown", f"{bad!r} -> {state}"
        assert 0.0 <= conf <= 1.0
    for flat in (0, 255, 128):
        state, _ = clf.classify(np.full((120, 60, 3), flat, np.uint8))
        assert state == "unknown", f"flat {flat} -> {state}"


@check("tiny and sliver crops do not crash and stay in range")
def test_tiny():
    clf = TrafficLightClassifier()
    for shape in ((10, 6, 3), (18, 12, 3), (40, 24, 3), (200, 6, 3)):
        img = cv2.resize(housing(), (shape[1], shape[0]))
        state, conf = clf.classify(img)
        assert state in ("red", "yellow", "green", "unknown"), state
        assert 0.0 <= conf <= 1.0, conf


@check("green lamp inside a cream housing is not read as amber")
def test_green_in_cream_housing():
    clf = TrafficLightClassifier()
    state, conf = clf.classify(glow_variant(body=CREAM, lamp=GREEN))
    assert state == "green", f"{state} ({conf:.2f}) - the housing won again"
    assert conf > 0


@check("green lamp wins when the housing is larger and more chromatic")
def test_not_largest_region():
    # Housing painted amber and lit green: the biggest chromatic region is the
    # paint, exactly the configuration that broke v1.1.
    clf = TrafficLightClassifier()
    img = glow_variant(body=(110, 200, 240), lamp=GREEN)
    state, conf = clf.classify(img)
    assert state == "green", f"{state} ({conf:.2f})"


@check("unlit housing abstains instead of naming amber")
def test_unlit_housing():
    clf = TrafficLightClassifier()
    state, conf = clf.classify(housing(body=CREAM, lit=()))
    assert state == "unknown", f"{state} ({conf:.2f})"
    # dark-bodied housing too, and a yellow-painted one lit nowhere
    state, _ = clf.classify(housing(body=(60, 70, 75), lit=()))
    assert state == "unknown", state


@check("amber lamp in a dark housing is named yellow")
def test_amber_lamp():
    clf = TrafficLightClassifier()
    state, conf = clf.classify(glow_variant(body=(55, 60, 65), lamp=AMBER, bg=(60, 65, 70)))
    assert state == "yellow", f"{state} ({conf:.2f})"


@check("amber lamp is never reported as red")
def test_amber_not_red():
    clf = TrafficLightClassifier()
    for body, bg in (((55, 60, 65), (60, 65, 70)), (CREAM, (90, 95, 100))):
        state, _ = clf.classify(glow_variant(body=body, lamp=AMBER, bg=bg))
        assert state != "red", f"amber read as red with body={body}"


@check("red lamp with a blown-out core is named red")
def test_red_blown_out():
    clf = TrafficLightClassifier()
    state, conf = clf.classify(glow_variant(body=(40, 42, 45), lamp=RED, bg=(25, 28, 32)))
    assert state == "red", f"{state} ({conf:.2f})"


@check("green lamp under strong glare is still found")
def test_glare():
    clf = TrafficLightClassifier()
    state, conf = clf.classify(glow_variant(body=(70, 75, 80), lamp=GREEN, bg=(210, 215, 220)))
    assert state == "green", f"{state} ({conf:.2f})"


@check("all three lamps lit at once abstains")
def test_ambiguous_all_lit():
    clf = TrafficLightClassifier()
    img = housing(body=(40, 45, 50), lit=(), bg=(30, 32, 36))
    for i, lamp in enumerate((RED, AMBER, GREEN)):
        r = int(min(img.shape) * 0.17)
        cy = int(img.shape[0] * (0.20 + 0.30 * i))
        cv2.circle(img, (img.shape[1] // 2, cy), int(r * 0.86), lamp, -1)
    state, conf = clf.classify(cv2.GaussianBlur(img, (0, 0), 1.5))
    assert state == "unknown", f"{state} ({conf:.2f}) - it should not pick one"


@check("lamp cut by the crop edge does not crash")
def test_edge_crop():
    clf = TrafficLightClassifier()
    full = glow_variant()
    for box in ((0, 0, 60, 300), (70, 150, 130, 300), (0, 240, 130, 300)):
        crop = full[box[1]:box[3], box[0]:box[2]]
        state, conf = clf.classify(crop)
        assert state in ("red", "yellow", "green", "unknown"), state
        assert 0.0 <= conf <= 1.0


@check("temporal voting waits, then publishes, and resets")
def test_temporal_voting():
    clf = TrafficLightClassifier()
    frame = glow_variant(body=CREAM, lamp=GREEN)
    seen = []
    for _ in range(clf.temporal_min_votes - 1):
        seen.append(clf.classify(frame, key="sig")[0])
    assert all(s == "unknown" for s in seen), seen
    for _ in range(3):
        state, _ = clf.classify(frame, key="sig")
    assert state == "green", state
    clf.reset_temporal()
    assert clf.classify(frame, key="sig")[0] == "unknown"


# ---------------------------------------------------------------- real crops
@check("real cream-housing crops: green lamp found, unlit abstains")
def test_real_cream_housing():
    clf = TrafficLightClassifier()
    # 39 and 26 are the two crops where v1.x reported amber on a lit green lamp.
    for idx, expected in ((26, "green"), (39, "green")):
        img = cv2.imread(str(ROOT / f"labels/traffic_lights/crops/{idx:02d}.jpg"))
        assert img is not None, idx
        state, _ = clf.classify(img)
        assert state == expected, f"crop {idx}: {state}"
    for idx in (3, 4, 5, 6, 28):
        img = cv2.imread(str(ROOT / f"labels/traffic_lights/crops/{idx:02d}.jpg"))
        state, _ = clf.classify(img)
        assert state == "unknown", f"unlit housing {idx} -> {state}"


@check("held-out v2 amber crops are not called red")
def test_v2_amber():
    clf = TrafficLightClassifier()
    for idx in (1, 2, 3):
        img = cv2.imread(str(ROOT / f"labels/traffic_lights/v2/crops/{idx:02d}.jpg"))
        if img is None:
            continue
        state, _ = clf.classify(img)
        assert state != "red", f"v2 crop {idx} (amber) -> {state}"


def main():
    failures = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failures)}/{len(RESULTS)} checks passed")
    for name, ok, err in failures:
        print(f"  FAILED {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
