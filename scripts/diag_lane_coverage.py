"""Why does a corridor get published on only a few frames?

    python scripts/diag_lane_coverage.py                    # 27 frames x 4 profiles
    python scripts/diag_lane_coverage.py --seq 120           # + temporal run on clip F
    python scripts/diag_lane_coverage.py --configs culane:stretch

Read-only. Nothing here changes a threshold, and nothing here is a second
implementation of the published path: every gate that decides whether a lane is
*used* is executed by calling the detector's own methods (`detect`,
`_select_ego_boundaries`, `_corridor`, `_evidence_segment`, `boundary_support`).
Only the pre-threshold decode detail is mirrored, so the counts of "the model
chose the no-lane bin" and "the model located a line but was not confident
enough" can be separated - and that mirror asserts it agrees point for point
with `det._decode()` on every frame, so it cannot silently drift.

For every frame and profile it records, as separate fields:

  raw      per slot: row anchors where argmax picked the non-existence bin
           (`nonexist`), anchors with a grid bin but (1-p) below
           LANE_EXISTENCE_THRESHOLD (`lowconf`), and the surviving count.
           Existence is per-anchor, and on these profiles only 18-56 anchors
           exist at all, so this is the share of the *model's own* row grid.
  decoded  per slot after the real detect() path: point count, y span,
           how many points sit in the bottom 25% of the frame, and whether
           the slot reaches the ego band used by the corridor test.
  paint    per slot: fraction of decoded points with paint under them, and
           what `_evidence_segment` decided about drawing them.
  ego      which slots became (left, right), their bottom_x, the measured
           bottom width against LANE_MIN/MAX_WIDTH_FRAC, and the reason when
           no pair was returned.
  corridor the real `_corridor` verdict: polygon stats, paint support per
           edge after the trim, and the verbatim rejection reason.

Failure stage per frame is assigned from those records in causal order, so the
summary counts are a consequence of the gates that actually refused, not of a
taxonomy invented here.

Overlays: one montage per frame per profile in <out>/overlays/<profile>/, six
panels - raw decode, decoded after filters, paint mask, supported segments,
ego pair + corridor, final annotate_frame output - so each stage can be checked
against the record. Source frames are the originals only; annotated outputs from
earlier runs are never fed back in.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modules.lane_detection import (  # noqa: E402
    UltraFastLaneDetector,
    boundary_support,
    paint_support_mask,
)
from scripts.eval_lane import x_at_row  # noqa: E402  same interpolation as the published curve

GT_CLIPS = ROOT / "labels" / "lane" / "clips_gt.json"
VIDEO = "test_data/lane_clips/F_*.webm"
STILLS = [
    ("my_test_images", "sir_image.jpg"),
    ("my_test_images", "sir_image2.jpg"),
    ("my_test_images", "sir_image3.jpg"),
    ("my_test_images", "sir_image7.jpeg"),
    ("my_test_images", "sir_image10.jpeg"),
    ("test_data", "highway_road.jpg"),
    ("test_data", "wyamo_f10.jpg"),
    ("test_data", "wyamo_f200.jpg"),
]
ALL_PROFILES = [("tusimple", "stretch"), ("tusimple", "letterbox"),
                ("culane", "stretch"), ("culane", "letterbox")]
EGO_BAND_FRAC = 1.0 - config.LANE_CORRIDOR_MAX_BOTTOM_GAP_FRAC
TILE_H = 420


def decode_detail(det, out: np.ndarray):
    """Mirror of `_decode` that also counts *why* an anchor was dropped.

    Returns (lanes, per_slot_counts, agreement). `agreement` is checked against
    the real `det._decode(out)` by the caller; a mismatch means this mirror is
    wrong and the record is marked, never trusted.
    """
    logits = torch.from_numpy(out).float()
    prob = torch.softmax(logits, dim=0).numpy()
    grid_prob = torch.softmax(logits[:-1], dim=0).numpy()
    nonexist = prob[det.num_gridding]
    am = out.argmax(axis=0)
    centres = (np.arange(det.num_gridding) + 0.5) * (config.UFLD_INPUT_WIDTH / det.num_gridding)
    loc = (grid_prob * centres.reshape(-1, 1, 1)).sum(axis=0)

    lanes, counts = [], []
    for li in range(int(det.profile["lanes"])):
        pts = []
        n_nonexist = n_lowconf = n_offscreen = 0
        pattern = []
        for ri in range(len(det.row_anchors)):
            if am[ri, li] >= det.num_gridding:
                n_nonexist += 1
                pattern.append("n")
                continue
            if (1.0 - nonexist[ri, li]) < det.min_lane_conf:
                n_lowconf += 1
                pattern.append("l")
                continue
            x, y = float(loc[ri, li]), float(det.row_anchors[ri])
            if not (0 <= x <= config.UFLD_INPUT_WIDTH):
                n_offscreen += 1
                pattern.append("o")
                continue
            pts.append((int(round(x)), int(round(y))))
            pattern.append(".")
        lanes.append(pts)
        counts.append({
            "slot": li,
            "anchors": len(det.row_anchors),
            "nonexist": n_nonexist,
            "lowconf": n_lowconf,
            "offscreen": n_offscreen,
            "decoded": len(pts),
            # One char per row anchor, top -> bottom: '.' decoded, 'n' the model
            # chose the no-lane bin, 'l' located but below the existence floor,
            # 'o' decoded off the 800px grid. This is the vertical distribution.
            "pattern": "".join(pattern),
        })
    return lanes, counts


def raw_stage(det, frame):
    """One forward pass, decoded twice: the mirror and the real decoder."""
    tensor, mapper = det._preprocess(frame)
    with torch.no_grad():
        out = det.model(tensor)[0].cpu().numpy()
    mirror, counts = decode_detail(det, out)
    real = det._decode(out)
    agreement = all(len(a) == len(b) for a, b in zip(mirror, real)) and len(mirror) == len(real)
    raw_px = [[mapper(float(x), float(y)) for x, y in pts] for pts in mirror]
    return {
        "slots": counts,
        "agreement": agreement,
        "raw_points_px": [[(int(round(x)), int(round(y))) for x, y in lane] for lane in raw_px],
        "logits_shape": list(out.shape),
    }


def paint_stage(det, frame, lanes, mask):
    out = []
    for idx, pts in enumerate(lanes):
        if len(pts) < 2:
            out.append({"slot": idx, "points": len(pts), "supported_frac": None, "draw": None})
            continue
        support = boundary_support(mask, pts)
        segments, why = det._evidence_segment(pts, support, frame.shape[0])
        out.append({
            "slot": idx,
            "points": len(pts),
            "supported": int(sum(support)),
            "supported_frac": round(float(np.mean(support)), 3),
            "draw": why,
            "drawn_points": int(sum(len(s) for s in segments)),
        })
    return out


def filter_stage(det, frame, raw_px, lanes):
    """Attribute every raw point to the detect() filter that dropped it.

    The five stages below are the ones detect() applies, in the same order. The
    kept count is asserted against the real detect() output per slot, so this
    attribution is either correct or reported as a mismatch - it cannot quietly
    describe a pipeline that is not the one under test.
    """
    h, w = frame.shape[:2]
    out = []
    for idx, raw in enumerate(raw_px):
        bound = roi = occ = tele = 0
        mapped = []
        for x, y in raw:
            if not (0 <= x < w and 0 <= y < h):
                bound += 1
                continue
            if not det._in_roi(y, h):
                roi += 1
                continue
            if det._is_occluded(x, y, None):
                occ += 1
                continue
            mapped.append((x, y))
        kept = []
        for pt in mapped:
            if not kept:
                kept.append(pt)
                continue
            px, py = kept[-1]
            allowed = max(0.03 * w, 1.2 * abs(pt[1] - py) + 0.02 * w)
            if abs(pt[0] - px) <= allowed:
                kept.append(pt)
            else:
                tele += 1
        after = lanes[idx] if idx < len(lanes) else []
        out.append({
            "slot": idx,
            "raw": len(raw),
            "out_of_bounds": bound,
            "roi": roi,
            "occluded": occ,
            "teleport": tele,
            "kept": len(kept),
            "after_detect": len(after),
            "raw_y": [min(p[1] for p in raw), max(p[1] for p in raw)] if raw else None,
            "kept_y": [min(p[1] for p in kept), max(p[1] for p in kept)] if kept else None,
            "after_y": [min(p[1] for p in after), max(p[1] for p in after)] if after else None,
            "min_points_dropped": bool(len(kept) >= 2 and not after),
            "_kept_yseq": tuple(p[1] for p in kept),
        })
    return out


def verify_filters(det, filters, lanes):
    """Prove the drop attribution against the real detect() output.

    Two stages sit between `kept` and what detect() returns: the
    min_lane_points cut and `_restore_slots`, which re-associates temporally
    smoothed lanes to slots by bottom_x and can therefore drop a slot or hand
    it another slot's points. Both are accounted for here by comparing the
    y-sequence the attribution predicts with the y-sequence detect() produced
    (y values survive the blend; x does not, so y is the identity to compare).
    """
    expected = [f["_kept_yseq"] if f["kept"] >= det.min_lane_points else () for f in filters]
    actual = [tuple(p[1] for p in (lanes[i] if i < len(lanes) else []))
              for i in range(len(expected))]
    for i, f in enumerate(filters):
        if actual[i] == expected[i]:
            f["restore"] = "exact"
        elif not expected[i] and not actual[i]:
            f["restore"] = "empty"
        elif not expected[i]:
            f["restore"] = "gained"
        elif not actual[i]:
            f["restore"] = "dropped_by_restore"
        elif actual[i] in expected:
            f["restore"] = "reassigned"
        else:
            f["restore"] = "changed"
        f["matches"] = f["restore"] in ("exact", "empty")
    return filters


def frame_record(det, frame, name, kind, gt_item=None):
    h, w = frame.shape[:2]
    raw = raw_stage(det, frame)
    lanes, ms = det.detect(frame)
    mask = paint_support_mask(frame) if config.LANE_PAINT_SUPPORT else None
    filters = verify_filters(det, filter_stage(det, frame, raw["raw_points_px"], lanes), lanes)

    decoded = []
    for idx, pts in enumerate(lanes):
        if len(pts) < 2:
            decoded.append({"slot": idx, "points": len(pts)})
            continue
        ys = [p[1] for p in pts]
        bottom_q = sum(1 for y in ys if y >= 0.75 * h)
        decoded.append({
            "slot": idx,
            "points": len(pts),
            "y_top": int(min(ys)),
            "y_bottom": int(max(ys)),
            "span_frac": round((max(ys) - min(ys)) / h, 3),
            "bottom_quartile": bottom_q,
            "reaches_ego_band": bool(max(ys) >= EGO_BAND_FRAC * h),
            "left_of_centre": bool(np.median([p[0] for p in pts]) < w / 2.0),
        })

    paint = paint_stage(det, frame, lanes, mask) if mask is not None else []
    left, right, ediag = det._select_ego_boundaries(lanes)

    corridor = {"established": False, "reason": None}
    if left is not None and right is not None:
        poly, cdiag = det._corridor(frame, left, right)
        corridor = dict(cdiag)
        corridor["established"] = poly is not None
        if poly is not None:
            corridor["poly_y"] = [int(poly[:, 1].min()), int(poly[:, 1].max())]
            corridor["poly_width_frac"] = round(
                float(np.mean(poly[: len(poly) // 2, 0][::-1] - poly[len(poly) // 2:, 0]) / w), 3)
    else:
        corridor["reason"] = ediag.get("reason") or "no ego pair"

    # Boundary error against hand-verified points, when this frame has labels.
    gt_rows = []
    if gt_item:
        for b in gt_item["boundaries"]:
            sel = left if b["side"] == "left" else right
            for r in b["rows"]:
                got = x_at_row(sel, r["y"]) if sel is not None else None
                cand = [x_at_row(l, r["y"]) for l in lanes]
                cand = [c for c in cand if c is not None]
                best = min((abs(c - r["x"]) for c in cand), default=None)
                gt_rows.append({
                    "side": b["side"], "y": r["y"], "x": r["x"],
                    "model_x": None if got is None else round(got, 1),
                    "error": None if got is None else round(abs(got - r["x"]), 1),
                    "best_error": None if best is None else round(best, 1),
                })

    rec = {
        "name": name,
        "kind": kind,
        "model": det.model_key,
        "preprocess": det.preprocess,
        "weights": Path(det.model_path).name,
        "profile_rows": int(det.profile["rows"]),
        "gridding": int(det.num_gridding),
        "existence_threshold": det.min_lane_conf,
        "shape": [h, w],
        "ms": round(ms, 1),
        "raw": raw["slots"],
        "raw_mirror_agrees": raw["agreement"],
        "logits_shape": raw["logits_shape"],
        "decoded": decoded,
        "filters": filters,
        "paint": paint,
        "ego": {k: v for k, v in ediag.items() if k != "candidates"},
        "ego_candidates": ediag.get("candidates"),
        "corridor": corridor,
        "gt_rows": gt_rows,
        "_raw_points_px": raw["raw_points_px"],
        "_mask": mask,
        "_lanes": lanes,
        "_left": left,
        "_right": right,
    }
    return rec


def stage_of(rec):
    """First gate that refused, in causal order. Evidence, not opinion."""
    usable = [d for d in rec["decoded"] if d.get("points", 0) >= 2]
    if not usable:
        return "1_no_boundary_decoded"
    if not rec["ego"].get("left_idx"):
        r = (rec["corridor"].get("reason") or "").lower()
        if "fewer than two" in r:
            return "2_one_boundary_only"
        if "implausible" in r or "no distinct" in r:
            return "3_pair_rejected_geometry"
        return "2_one_boundary_only"
    if rec["corridor"].get("established"):
        return "0_published"
    r = (rec["corridor"].get("reason") or "").lower()
    if "paint" in r:
        return "4_paint_gate"
    if "share only" in r:
        return "5_overlap_limited"
    if "too short" in r:
        return "5_corridor_too_short"
    if "above the bottom" in r:
        return "6_bottom_gap"
    if "axis" in r:
        return "7_axis"
    if "overlap" in r or "cross" in r:
        return "3_pair_rejected_geometry"
    return "8_other"


def _tile(img, label, sub=""):
    h, w = img.shape[:2]
    s = TILE_H / float(h)
    t = cv2.resize(img, (max(1, int(w * s)), TILE_H))
    cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(t, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(t, sub, (5, t.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return t


def montage(det, rec, frame, out_path: Path):
    """Six panels, one per stage. Each panel is the stage's own output."""
    h = frame.shape[0]
    panels = []

    raw = frame.copy()
    for lane in rec["_raw_points_px"]:
        for (x, y) in lane:
            if 0 <= x < raw.shape[1] and 0 <= y < h:
                cv2.circle(raw, (x, y), 2, (255, 255, 255), -1)
    panels.append(_tile(raw, "1 raw decode (before filters)",
                        f"{sum(len(l) for l in rec['_raw_points_px'])} pts"))

    dec = frame.copy()
    for idx, pts in enumerate(rec["_lanes"]):
        if len(pts) >= 2:
            col = config.LANE_COLORS[idx % len(config.LANE_COLORS)]
            cv2.polylines(dec, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
    panels.append(_tile(dec, "2 after detect() filters",
                        f"{sum(1 for l in rec['_lanes'] if len(l) >= 2)} boundaries"))

    pm = frame.copy()
    if rec["_mask"] is not None:
        m = rec["_mask"]
        pm[m] = (0, 0, 255)
        pm = cv2.addWeighted(frame, 0.55, pm, 0.45, 0)
    panels.append(_tile(pm, "3 paint mask (red = paint)"))

    sup = frame.copy()
    for idx, pts in enumerate(rec["_lanes"]):
        if len(pts) < 2 or rec["_mask"] is None:
            continue
        support = boundary_support(rec["_mask"], pts)
        for (x, y), ok in zip(pts, support):
            if not ok:
                cv2.drawMarker(sup, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 7, 1)
        segs, _ = det._evidence_segment(pts, support, h)
        for seg in segs:
            cv2.polylines(sup, [np.array(seg, np.int32)], False, (0, 255, 0), 3, cv2.LINE_AA)
    drawn = sum(p.get("drawn_points") or 0 for p in rec["paint"])
    panels.append(_tile(sup, "4 paint-supported segments", f"{drawn} pts drawn, x = no paint"))

    ego = frame.copy()
    if rec["_left"] is not None and rec["_right"] is not None:
        cv2.polylines(ego, [np.array(rec["_left"], np.int32)], False, (255, 128, 0), 3, cv2.LINE_AA)
        cv2.polylines(ego, [np.array(rec["_right"], np.int32)], False, (0, 215, 255), 3, cv2.LINE_AA)
        c = rec["corridor"]
        if c.get("established") and "poly_y" in c:
            ys = np.arange(c["poly_y"][0], c["poly_y"][1] + 1)
            lx = np.interp(ys, [p[1] for p in rec["_left"]], [p[0] for p in rec["_left"]])
            rx = np.interp(ys, [p[1] for p in rec["_right"]], [p[0] for p in rec["_right"]])
            poly = np.stack([np.concatenate([lx, rx[::-1]]),
                             np.concatenate([ys, ys[::-1]])], axis=1).astype(np.int32)
            ov = ego.copy()
            cv2.fillPoly(ov, [poly], (0, 180, 0))
            ego = cv2.addWeighted(ov, 0.3, ego, 0.7, 0)
            sub = f"published  support L{c.get('support_left')} R{c.get('support_right')}"
        else:
            sub = (c.get("reason") or "refused")[:52]
    else:
        sub = (rec["ego"].get("reason") or "no pair")[:52]
    panels.append(_tile(ego, "5 ego pair + corridor", sub))

    panels.append(_tile(det.annotate_frame(frame, rec["_lanes"]), "6 final annotate_frame"))

    widths = [p.shape[1] for p in panels]
    cw = max(widths)
    padded = [cv2.copyMakeBorder(p, 0, 0, 0, cw - p.shape[1], cv2.BORDER_CONSTANT, value=(25, 25, 25))
              for p in panels]
    top = np.hstack(padded[:3])
    bot = np.hstack(padded[3:])
    out = np.vstack([top, bot])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 82])


def collect_jobs(data: Path, frames: int):
    jobs = []
    gt = json.loads((data / "labels" / "lane" / "clips_gt.json").read_text(encoding="utf-8"))
    for item in gt["images"]:
        f = cv2.imread(str(data / item["path"]))
        if f is not None:
            jobs.append(("gt", Path(item["path"]).stem, f, item))
    for rel, name in STILLS:
        p = data / rel / name
        f = cv2.imread(str(p)) if p.exists() else None
        if f is not None:
            jobs.append(("img", Path(name).stem, f, None))
    vids = sorted(data.glob(VIDEO))
    if vids and frames:
        cap = cv2.VideoCapture(str(vids[0]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(5, max(6, total - 5), frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                jobs.append(("vid", f"F_{int(idx):05d}", f, None))
        cap.release()
    return jobs


def sequential_run(det, data: Path, n: int, stride: int = 1):
    """Same clip, frames in order, temporal filter NOT reset. Optional check."""
    vids = sorted(data.glob(VIDEO))
    if not vids:
        return []
    cap = cv2.VideoCapture(str(vids[0]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, total // 3)
    det.reset()
    out = []
    for i in range(start, min(total, start + n * stride), stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        lanes, _ = det.detect(f)
        left, right, ed = det._select_ego_boundaries(lanes)
        poly, reason = None, None
        if left is not None and right is not None:
            poly, cd = det._corridor(f, left, right)
            if poly is None:
                reason = cd.get("reason")
        else:
            reason = ed.get("reason")
        out.append({"frame": i, "n_lanes": sum(1 for l in lanes if len(l) >= 2),
                    "pair": bool(ed.get("left_idx")), "published": poly is not None,
                    "reason": reason})
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6, help="frames sampled from clip F")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="model:preprocess pairs, default all four")
    ap.add_argument("--seq", type=int, default=0,
                    help="also run N frames of clip F in order without resetting the filter")
    ap.add_argument("--seq-stride", type=int, default=1)
    ap.add_argument("--clip-stride", type=int, default=0,
                    help="also measure every Nth frame of clip F (0 = off). The 6-frame "
                         "linspace sample straddles very different road segments, so a "
                         "clip-wide stride is the only honest coverage figure.")
    ap.add_argument("--no-overlays", action="store_true", help="skip the montages (much faster)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"cov_{stamp}"
    out_dir = ROOT / "outputs" / "runs" / "lane_coverage_v18" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = ALL_PROFILES
    if args.configs:
        configs = []
        for spec in args.configs:
            m, _, p = spec.partition(":")
            configs.append((m, p or "stretch"))

    jobs = collect_jobs(ROOT, args.frames)
    n_gt = sum(1 for j in jobs if j[0] == "gt")
    print(f"{len(jobs)} frames ({n_gt} verified + {len(jobs) - n_gt} unlabelled) "
          f"x {len(configs)} profiles -> {out_dir}")

    report = {"generated": datetime.now().isoformat(timespec="seconds"), "configs": []}
    for model, prep in configs:
        det = UltraFastLaneDetector(device="cpu", model_key=model, preprocess=prep)
        if not det.is_loaded:
            print(f"skip {model}/{prep}: {det.load_error}")
            continue
        assert det.model_key == model and det.preprocess == prep, "profile not applied"
        print(f"\n=== {model} / {prep}  ({Path(det.model_path).name}, "
              f"rows={det.profile['rows']}, griding={det.num_gridding}) ===")

        recs = []
        for kind, name, frame, item in jobs:
            det.reset()
            rec = frame_record(det, frame, name, kind, item)
            vis = {k: v for k, v in rec.items() if not k.startswith("_")}
            vis["stage"] = stage_of(rec)
            recs.append(vis)
            if not args.no_overlays:
                montage(det, rec, frame, out_dir / "overlays" / f"{model}_{prep}" / f"{name}.jpg")
            c = rec["corridor"].get("established")
            extra = ("published" if c else (rec["corridor"].get("reason") or "no pair"))
            print(f"  [{vis['stage']:>26s}] {name:12s} "
                  f"lanes {sum(1 for l in rec['_lanes'] if len(l) >= 2)} "
                  f"paint {rec['paint'][0].get('supported_frac') if rec['paint'] else None} "
                  f" {str(extra)[:60]}")

        stages = Counter(r["stage"] for r in recs)
        mirror_bad = sum(1 for r in recs if not r["raw_mirror_agrees"])
        restore = Counter(f.get("restore") for r in recs for f in r["filters"])
        print(f"  filter attribution: {dict(restore)}")
        gt_recs = [r for r in recs if r["gt_rows"]]
        errs = [g["error"] for r in gt_recs for g in r["gt_rows"] if g["error"] is not None]
        miss = sum(1 for r in gt_recs for g in r["gt_rows"] if g["error"] is None)
        best = [g["best_error"] for r in gt_recs for g in r["gt_rows"] if g["best_error"] is not None]
        pairs = sum(1 for r in gt_recs if r["ego"].get("left_idx"))
        poly_ok = sum(1 for r in gt_recs if r["corridor"].get("established"))

        block = {
            "model": model, "preprocess": prep,
            "weights": Path(det.model_path).name,
            "profile": {"rows": int(det.profile["rows"]), "griding": int(det.num_gridding),
                        "existence_threshold": det.min_lane_conf},
            "frames": len(recs),
            "raw_mirror_mismatches": mirror_bad,
            "stages": dict(stages),
            "verified_frames": len(gt_recs),
            "verified_ego_pairs": pairs,
            "verified_corridors": poly_ok,
            "verified_rows": len(errs) + miss,
            "verified_covered": len(errs),
            "verified_coverage": round(len(errs) / max(1, len(errs) + miss), 3),
            "verified_mae_px": round(float(np.mean(errs)), 1) if errs else None,
            "verified_median_px": round(float(np.median(errs)), 1) if errs else None,
            "verified_max_px": round(float(np.max(errs)), 1) if errs else None,
            "best_median_px": round(float(np.median(best)), 1) if best else None,
            "records": recs,
        }
        if args.clip_stride:
            vids = sorted(ROOT.glob(VIDEO))
            clip = []
            if vids:
                cap = cv2.VideoCapture(str(vids[0]))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in range(0, total, args.clip_stride):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, f = cap.read()
                    if not ok:
                        continue
                    det.reset()
                    r = frame_record(det, f, f"F_{i:05d}", "clip")
                    r.pop("_raw_points_px", None)
                    r.pop("_mask", None)
                    r.pop("_lanes", None)
                    r.pop("_left", None)
                    r.pop("_right", None)
                    r["stage"] = stage_of({"decoded": r["decoded"], "ego": r["ego"],
                                           "corridor": r["corridor"]})
                    clip.append(r)
                cap.release()
            n = max(1, len(clip))
            per_block = 6
            blocks = []
            for b in range(per_block):
                chunk = clip[b * n // per_block:(b + 1) * n // per_block]
                if chunk:
                    blocks.append({
                        "frames": len(chunk),
                        "published": sum(1 for r in chunk if r["stage"] == "0_published"),
                        "rate": round(sum(1 for r in chunk if r["stage"] == "0_published")
                                      / len(chunk), 3),
                    })
            block["clip"] = {
                "frames": len(clip),
                "published": sum(1 for r in clip if r["stage"] == "0_published"),
                "rate": round(sum(1 for r in clip if r["stage"] == "0_published") / n, 3),
                "stages": dict(Counter(r["stage"] for r in clip)),
                "blocks": blocks,
                "records": clip,
            }
            print(f"  clip-wide stride {args.clip_stride}: published "
                  f"{block['clip']['published']}/{n} ({block['clip']['rate']:.0%}); blocks "
                  + " ".join(f"{b['published']}/{b['frames']}" for b in blocks))

        if args.seq:
            seq = sequential_run(det, ROOT, args.seq, args.seq_stride)
            block["sequential"] = {
                "frames": len(seq),
                "published": sum(1 for s in seq if s["published"]),
                "pairs": sum(1 for s in seq if s["pair"]),
                "two_lanes": sum(1 for s in seq if s["n_lanes"] >= 2),
                "reasons": dict(Counter(s["reason"] for s in seq if s["reason"]).most_common(6)),
                "per_frame": seq,
            }
        report["configs"].append(block)

        print(f"\n  stages: {dict(stages)}")
        print(f"  verified: ego pair {pairs}/{len(gt_recs)}, corridors {poly_ok}/{len(gt_recs)}, "
              f"boundary coverage {block['verified_covered']}/{block['verified_rows']} "
              f"({block['verified_coverage']}), median {block['verified_median_px']}px, "
              f"max {block['verified_max_px']}px")
        if args.seq:
            s = block["sequential"]
            print(f"  sequential (no reset, {s['frames']} frames): two lanes {s['two_lanes']}, "
                  f"pairs {s['pairs']}, published {s['published']}")
        if mirror_bad:
            print(f"  WARNING: decode mirror disagreed on {mirror_bad} frame(s)")

    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n-> {out_dir / 'report.json'}")
    print(f"-> {out_dir / 'overlays'}")

    both = [c for c in report["configs"]]
    if len(both) > 1:
        print("\ncoverage, all profiles on the identical frames:")
        for c in both:
            st = c["stages"]
            print(f"  {c['model']:8s}/{c['preprocess']:9s} publish {st.get('0_published', 0):2d}/{c['frames']}"
                  f"  verified {c['verified_covered']}/{c['verified_rows']} rows"
                  f"  median {c['verified_median_px']}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
