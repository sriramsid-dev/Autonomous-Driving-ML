"""
Corridor-validity tests. Runs without pytest and without model weights:

    python tests/test_lane_corridor.py

Every case builds a synthetic 640x480 frame with known paint, then feeds the
corridor builder boundary polylines that reproduce a failure seen in the real
runs. The detector is constructed with a missing checkpoint on purpose: none of
these paths touch the network, and that also proves the corridor logic cannot
silently depend on inference.

Cases, each named after the real failure it reproduces:

  field spill          frame 20/40 of clip F - right edge in the grass
  hillside lane        frame 20 of clip F - left edge up the bank
  unsupported tail     a boundary that keeps going after the paint stops
  overlap-limited       two edges that only co-exist over a few rows
  paint-shortened       ample overlap, but the paint only covers part of it
  missing boundary     only one side decoded
  hood phantom         bottom rows on the dark bonnet, not the road
  implausible width    two edges far wider apart than any lane
  floating corridor    both edges painted but nowhere near the vehicle
  gap handling         one missed sample in the middle of a solid line

Exit code is non-zero if any check fails. Nothing here asserts a metric.
"""

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modules.lane_detection import (  # noqa: E402
    UltraFastLaneDetector,
    boundary_support,
    close_short_gaps,
    paint_support_mask,
)

RESULTS = []
H, W = 480, 640
PAINT = (225, 225, 225)
ASPHALT = (88, 88, 88)
GRASS = (60, 96, 66)


def check(name):
    def deco(fn):
        def wrapper():
            try:
                fn()
                RESULTS.append((name, True, ""))
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()

        return wrapper

    return deco


def base_frame(road_right=470, bonnet_top=None, width=None):
    """Asphalt trapezoid, grass either side, dark bonnet strip at the bottom."""
    w = width or W
    frame = np.zeros((H, w, 3), np.uint8)
    frame[:] = GRASS
    cv2.fillPoly(frame, [np.array([[150, H], [road_right, H],
                                   [road_right - 90, 200], [235, 200]], np.int32)], ASPHALT)
    frame[H - 26 if bonnet_top is None else bonnet_top:] = (18, 18, 20)  # bonnet
    return frame


def draw_paint(frame, pts, colour=PAINT, thickness=7, dx=0):
    arr = np.array([[int(x + dx), int(y)] for x, y in pts], np.int32)
    cv2.polylines(frame, [arr], False, colour, thickness)


def left_pts(y_top=250, y_bot=452, n=12):
    ys = np.linspace(y_top, y_bot, n)
    xs = np.linspace(300, 196, n)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def right_pts(y_top=250, y_bot=452, n=12):
    ys = np.linspace(y_top, y_bot, n)
    xs = np.linspace(322, 448, n)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def detector():
    """A detector that never loads weights - these paths do not need the net."""
    det = UltraFastLaneDetector(model_path=str(ROOT / "models" / "__missing__.pth"))
    assert not det.is_loaded
    return det


def corridor_of(det, frame, lanes):
    det.annotate_frame(frame, lanes)
    return det.last_corridor_established, dict(det.last_corridor_diagnostics)


@check("paint mask finds the painted lines and not the grass")
def test_mask():
    frame = base_frame()
    draw_paint(frame, left_pts())
    draw_paint(frame, right_pts())
    mask = paint_support_mask(frame)
    assert boundary_support(mask, left_pts()) == [True] * len(left_pts())
    assert boundary_support(mask, right_pts()) == [True] * len(right_pts())
    grass_only = boundary_support(mask, [(560, y) for y in (300, 350, 400)])
    assert not any(grass_only), grass_only


@check("a good pair of painted edges yields a corridor")
def test_happy_path():
    frame = base_frame()
    draw_paint(frame, left_pts())
    draw_paint(frame, right_pts())
    det = detector()
    ok, diag = corridor_of(det, frame, [left_pts(), right_pts()])
    assert ok, diag
    assert diag.get("support_both", 0) == 1.0, diag
    assert diag.get("bottom_gap_frac", 1) <= config.LANE_CORRIDOR_MAX_BOTTOM_GAP_FRAC, diag


@check("field spill: right edge in the grass is refused (clip F frames 20/40)")
def test_field_spill():
    frame = base_frame()
    left = left_pts()
    right = right_pts()
    draw_paint(frame, left)
    draw_paint(frame, right)
    # the decoder's right edge wanders off the paint into the field
    phantom = [(x + 120, y) for x, y in right]
    det = detector()
    ok, diag = corridor_of(det, frame, [left, phantom])
    assert not ok, diag
    assert "paint" in (diag.get("reason") or "") or "short" in (diag.get("reason") or ""), diag


@check("hillside lane: an edge that leaves the paint is not drawn")
def test_hillside_lane():
    frame = base_frame()
    left = left_pts()
    draw_paint(frame, left)
    # upper half of the trace runs up the bank, well left of the paint
    bank = [(-100 + x, y) for x, y in left[:6]]
    bank = [(2, y) for _, y in bank]
    trace = bank + left[6:]
    det = detector()
    det.annotate_frame(frame, [trace])
    draw = det.last_corridor_diagnostics.get("lane_draw", {})
    assert draw.get(0, "").startswith("drawn"), draw
    assert det.last_corridor_diagnostics["lane_support"][0]["supported"] < len(trace)


@check("unsupported tail: corridor top stops where the paint stops")
def test_unsupported_tail():
    frame = base_frame()
    left = left_pts(y_top=330)   # paint only in the lower part of the frame
    right = right_pts(y_top=330)
    draw_paint(frame, left)
    draw_paint(frame, right)
    # both decoded traces keep going up past the end of the paint
    tail = [((x, y - 80)) for x, y in [(300, 330)]] + [((290, 250)), ((285, 210)), ((280, 180))]
    left_ext = tail + left
    right_ext = [((x + 30, y)) for x, y in tail] + right
    det = detector()
    ok, diag = corridor_of(det, frame, [left_ext, right_ext])
    assert ok, diag
    assert diag["y_lo"] >= 325, diag           # corridor starts at the paint
    assert diag.get("trimmed_rows", 0) > 0, diag


@check("overlap-limited: refusal names the overlap, not the paint test")
def test_overlap_limited():
    # The right edge appears only in the bottom 52 rows, and the paint covers
    # every one of them. Nothing is trimmed: the two edges simply never share
    # the 58 rows a corridor needs. That is a checkpoint coverage limit, so the
    # reason must not claim the paint filter shorted the corridor.
    frame = base_frame()
    left = left_pts()
    right = [(int(x), int(y)) for x, y in zip(np.linspace(340, 448, 6),
                                              np.linspace(400, 452, 6))]
    draw_paint(frame, left)
    draw_paint(frame, right)
    det = detector()
    ok, diag = corridor_of(det, frame, [left, right])
    assert not ok, diag
    reason = diag.get("reason") or ""
    assert "share only" in reason and "need" in reason, diag
    assert "supported corridor too short" not in reason, diag
    assert diag["overlap_rows"] == 52, diag
    assert "trimmed_rows" not in diag, diag     # the paint did not cut anything


@check("paint-shortened: refusal still names the paint trim")
def test_paint_shortened():
    # Both edges span 202 rows, so overlap is ample; only the bottom 37 rows
    # carry paint. Here the trim really is the binding constraint, and the
    # reason has to keep saying so.
    frame = base_frame()
    left = left_pts()
    right = right_pts()
    draw_paint(frame, [(int(x), int(y)) for x, y in zip(np.linspace(215, 196, 6),
                                                        np.linspace(415, 452, 6))])
    draw_paint(frame, [(int(x), int(y)) for x, y in zip(np.linspace(425, 448, 6),
                                                        np.linspace(415, 452, 6))])
    det = detector()
    ok, diag = corridor_of(det, frame, [left, right])
    assert not ok, diag
    reason = diag.get("reason") or ""
    assert "supported corridor too short" in reason, diag
    assert "share only" not in reason, diag
    assert diag["overlap_rows"] > config.LANE_CORRIDOR_MIN_ROWS_FRAC * H, diag
    assert diag.get("trimmed_rows", 0) > 0, diag


@check("too little shared span is named before any sampling")
def test_tiny_overlap():
    # 12 shared rows is under the 40 needed to interpolate a grid at all; the
    # old code reported this as "do not overlap or cross", which hid the fact
    # that both boundaries exist and merely co-exist over too short a stretch.
    frame = base_frame()
    left = left_pts()
    right = [(int(x), int(y)) for x, y in zip(np.linspace(430, 448, 6),
                                              np.linspace(440, 452, 6))]
    det = detector()
    ok, diag = corridor_of(det, frame, [left, right])
    assert not ok, diag
    reason = diag.get("reason") or ""
    assert "share only 12 rows" in reason, diag
    assert "do not overlap or cross" not in reason, diag
    assert diag["overlap_rows"] == 12, diag
    # Edges that are genuinely disjoint are a different failure and keep the
    # wording they always had.
    far = [(int(x), int(y)) for x, y in zip(np.linspace(430, 448, 6),
                                            np.linspace(120, 200, 6))]
    ok2, diag2 = corridor_of(det, frame, [left, far])
    assert not ok2, diag2
    assert "do not overlap or cross" in (diag2.get("reason") or ""), diag2
    assert diag2["overlap_rows"] == 0, diag2


@check("missing boundary: one side is not a corridor")
def test_missing_boundary():
    frame = base_frame()
    left = left_pts()
    draw_paint(frame, left)
    det = detector()
    ok, diag = corridor_of(det, frame, [left])
    assert not ok
    assert "two usable boundaries" in (diag.get("reason") or ""), diag


@check("hood phantom: bottom rows on the bonnet are trimmed, not painted")
def test_hood():
    bonnet = 430
    frame = base_frame(bonnet_top=bonnet)
    det = detector()
    left = left_pts(y_bot=424)
    right = right_pts(y_bot=424)
    draw_paint(frame, left)
    draw_paint(frame, right)
    # both traces keep going onto the dark bonnet
    left_h = left + [((int(x), y)) for x, y in zip(np.linspace(196, 186, 6), np.linspace(434, 476, 6))]
    right_h = right + [((int(x), y)) for x, y in zip(np.linspace(448, 454, 6), np.linspace(434, 476, 6))]
    ok, diag = corridor_of(det, frame, [left_h, right_h])
    assert ok, diag
    assert diag["y_hi"] <= bonnet + 4, diag          # 5-row support window tolerance
    assert diag["bottom_gap_frac"] > 0.08, diag      # well clear of the bonnet rows
    support = boundary_support(paint_support_mask(frame), left_h)
    assert not any(support[-5:]), support            # the bonnet rows really are unsupported


@check("hood-only evidence: a lane that exists only on the bonnet is refused")
def test_hood_only():
    frame = base_frame()
    det = detector()
    hood = [((int(x), y)) for x, y in zip(np.linspace(200, 440, 8), np.linspace(462, 478, 8))]
    ok, diag = corridor_of(det, frame, [hood, [(x + 8, y) for x, y in hood]])
    assert not ok, diag


@check("implausible width: edges 80% of the frame apart are refused")
def test_implausible_width():
    frame = base_frame()
    det = detector()
    a = left_pts()
    b = [(int(x) + 500, y) for x, y in right_pts()]  # straight off to the right
    draw_paint(frame, a)
    ok, diag = corridor_of(det, frame, [a, b])
    assert not ok
    assert "width" in (diag.get("reason") or ""), diag


@check("off-axis corridor is refused now that the axis slack is 0")
def test_axis_slack():
    # On clip F the camera axis sits ~300px right of a CORRECT corridor's
    # midpoint, so a 0.25-width slack was tried and admitted a corridor covering
    # both lanes (frame 721). This pins the decision: painted edges whose bottom
    # interval ends 30px left of the camera axis are refused, even though a 0.25
    # slack would have accepted them.
    frame = base_frame(road_right=880, width=900)
    det = detector()
    ys = np.linspace(250, 452, 12)
    left = [(int(x), int(y)) for x, y in zip(np.linspace(300, 180, 12), ys)]
    right = [(int(x), int(y)) for x, y in zip(np.linspace(500, 420, 12), ys)]
    draw_paint(frame, left)
    draw_paint(frame, right)
    ok, diag = corridor_of(det, frame, [left, right])
    assert not ok, diag
    assert "camera axis" in (diag.get("reason") or ""), diag
    slack = 0.25 * (right[-1][0] - left[-1][0]) + right[-1][0]
    assert slack > frame.shape[1] / 2, (slack, frame.shape[1] / 2)  # the old rule


@check("a corridor the camera axis falls inside is accepted")
def test_axis_inside():
    frame = base_frame()
    det = detector()
    draw_paint(frame, left_pts())
    draw_paint(frame, right_pts())
    ok, diag = corridor_of(det, frame, [left_pts(), right_pts()])
    assert ok, diag
    assert abs(diag.get("axis_offset_px", 999)) < W / 2, diag


@check("floating corridor: painted but nowhere near the vehicle is refused")
def test_floating():
    frame = base_frame()
    det = detector()
    top = left_pts(y_top=205, y_bot=300)
    top_r = right_pts(y_top=205, y_bot=300)
    draw_paint(frame, top)
    draw_paint(frame, top_r)
    ok, diag = corridor_of(det, frame, [top, top_r])
    assert not ok, diag
    assert "lane this vehicle is in" in (diag.get("reason") or ""), diag


@check("gap handling: one missed sample does not split a solid line")
def test_gaps():
    assert close_short_gaps([1, 0, 1], 1)[1] is True
    flags = [True, True, False, True, True]
    assert close_short_gaps(flags, 1) == [True, True, True, True, True]
    assert close_short_gaps([True, False, False, True], 1) == [True, False, False, True]


@check("corridor is not published when the ego band is empty")
def test_empty_ego_band():
    partial = [(x, y) for x, y in left_pts(y_top=210, y_bot=300)]
    seg, why = UltraFastLaneDetector._evidence_segment(partial, [True] * len(partial), H)
    assert seg == [] and "ego rows" in why, (seg, why)


@check("drawn boundaries are reported when nothing is rendered")
def test_reasons_reported():
    frame = base_frame()
    det = detector()
    det.annotate_frame(frame, [[(1, 1)], []] * 2)
    diag = det.last_corridor_diagnostics
    assert "reason" in diag, diag


def main() -> int:
    print(f"config: paint_support={config.LANE_PAINT_SUPPORT} "
          f"ego_anchor={config.LANE_REQUIRE_EGO_ANCHOR} "
          f"max_bottom_gap={config.LANE_CORRIDOR_MAX_BOTTOM_GAP_FRAC} "
          f"min_rows={config.LANE_CORRIDOR_MIN_ROWS_FRAC}")
    for fn in (test_mask, test_happy_path, test_field_spill, test_hillside_lane,
               test_unsupported_tail, test_overlap_limited, test_paint_shortened,
               test_tiny_overlap, test_missing_boundary, test_hood, test_hood_only,
               test_implausible_width, test_axis_slack, test_axis_inside,
               test_floating, test_gaps, test_empty_ego_band, test_reasons_reported):
        fn()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    for name, ok, err in RESULTS:
        if not ok:
            print(f"  FAILED {name}: {err}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
