"""
Regression suite. Runs without pytest:

    python tests/test_regression.py
    python tests/test_regression.py --video "D:/downloads/Wyamo Car video.mp4"

Checks that every module initialises, images and video process end to end,
tracker state resets between sequences, and the documented CLI options still
work. It asserts *behaviour*, never accuracy: a pass means "did what it is
supposed to do", not "detected the right things".

Exit code is non-zero if any check fails.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

RESULTS = []


def check(name):
    def deco(fn):
        def wrapper(ctx):
            try:
                fn(ctx)
                RESULTS.append((name, True, ""))
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            return RESULTS

        return wrapper

    return deco


@check("config paths and weights exist")
def test_paths(ctx):
    assert Path(config.YOLO_MODEL_PATH).exists(), config.YOLO_MODEL_PATH
    for key, prof in config.LANE_PROFILES.items():
        assert Path(prof["weights"]).exists(), f"{key}: {prof['weights']}"


@check("lane checkpoints load strictly for every profile")
def test_lane_profiles(ctx):
    from modules.lane_detection import UltraFastLaneDetector

    for key in config.LANE_PROFILES:
        det = UltraFastLaneDetector(device="cpu", model_key=key)
        assert det.is_loaded, f"{key} failed to load: {det.load_error}"
        assert det.profile["rows"] == len(det.row_anchors)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        lanes, _ = det.detect(frame)
        assert len(lanes) == det.profile["lanes"], "wrong number of lane slots"


@check("traffic-light classifier rejects blank and tiny crops")
def test_tl_reject(ctx):
    from modules.traffic_light_classification import TrafficLightClassifier

    clf = TrafficLightClassifier()
    state, conf = clf.classify(np.zeros((80, 40, 3), dtype=np.uint8))
    assert state == "unknown" and conf == 0.0, (state, conf)
    state, _ = clf.classify(np.zeros((4, 3, 3), dtype=np.uint8))
    assert state == "unknown"


@check("image pipeline: all modules, module subsets, crop mode")
def test_images(ctx):
    from main import PerceptionPipeline

    pipe = PerceptionPipeline(device="cpu")
    img = ROOT / "test_data" / "green_tl.jpg"
    out = ctx["tmp"] / "all.jpg"
    ann, telem = pipe.process_image(img, str(out), modules="all")
    assert out.exists() and cv2.imread(str(out)) is not None
    assert telem["modules_executed"]["module_B_traffic_lights"] is True
    assert telem["resolution"] == "1200x1600"

    for mods in ("A", "B", "C"):
        o = ctx["tmp"] / f"{mods}.jpg"
        _, t = pipe.process_image(img, str(o), modules=mods)
        assert o.exists()
        flags = t["modules_executed"]
        expected = {"A": "module_A_object_tracking", "B": "module_B_traffic_lights",
                    "C": "module_C_lane_detection"}[mods]
        assert flags[expected] is True
        assert sum(1 for k, v in flags.items() if v) == 1, (mods, flags)

    crop = ROOT / "test_data" / "tl_crop_green.jpg"
    o = ctx["tmp"] / "crop.jpg"
    _, t = pipe.process_image(crop, str(o), modules="B", is_crop=True)
    assert o.exists() and t["detection_counts"]["traffic_lights"] == 1


@check("missing image raises a clear error")
def test_missing_image(ctx):
    from main import PerceptionPipeline

    pipe = PerceptionPipeline(enable_objects=False, enable_traffic_lights=True, enable_lanes=False)
    try:
        pipe.process_image(str(ROOT / "test_data" / "does_not_exist.jpg"))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


@check("video pipeline writes a playable file and resets the tracker")
def test_video(ctx):
    video = ctx["video"]
    if not video or not Path(video).exists():
        raise AssertionError("no video available (pass --video)")
    from main import PerceptionPipeline
    from modules.object_detection_tracking import ObjectDetectionTracker

    out = ctx["tmp"] / "clip.mp4"
    pipe = PerceptionPipeline(device="cpu")
    summary = pipe.process_video(str(video), str(out), max_frames=5, save_keyframes=False)
    assert out.exists() and out.stat().st_size > 0
    assert summary["frames_processed"] == 5

    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    cap.release()

    tracker = ObjectDetectionTracker(device="cpu")
    cap = cv2.VideoCapture(str(video))
    ids_first = set()
    for _ in range(4):
        ok, frame = cap.read()
        assert ok
        objs, _ = tracker.track(frame, persist=True)
        ids_first |= {o["track_id"] for o in objs if o["track_id"] is not None}
    tracker.reset_tracker()
    ids_second = set()
    for _ in range(4):
        ok, frame = cap.read()
        assert ok
        objs, _ = tracker.track(frame, persist=True)
        ids_second |= {o["track_id"] for o in objs if o["track_id"] is not None}
    cap.release()
    assert ids_first and ids_second, "no tracks produced"
    assert min(ids_second) == 1, f"IDs did not restart at 1 after reset: {sorted(ids_second)}"
    assert not (ids_first & ids_second) or min(ids_second) == 1


@check("CLI options still work (imgsz, lane-model, lane-preprocess, crop, output)")
def test_cli(ctx):
    img = ROOT / "test_data" / "highway_road.jpg"
    out = ctx["tmp"] / "cli.jpg"
    cmd = [sys.executable, "main.py", "--image", str(img), "-m", "C",
           "--imgsz", "480", "--lane-model", "culane", "--lane-preprocess", "letterbox",
           "--output-image", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    assert out.exists()
    # --lane-model used to be ignored because the detector bound its defaults at
    # import time; assert the requested profile is the one that actually loaded.
    assert "UFLD 'culane'" in r.stdout, r.stdout[-800:]
    assert "culane_18.pth" in r.stdout, r.stdout[-800:]

    out2 = ctx["tmp"] / "cli_crop.jpg"
    cmd = [sys.executable, "main.py", "--image", str(ROOT / "test_data" / "tl_crop_red.jpg"),
           "--crop", "-m", "B", "-o", str(out2)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    assert out2.exists()

    r = subprocess.run([sys.executable, "main.py", "--image", "nope.jpg"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0, "missing image should exit non-zero"


@check("evaluation harness runs on the committed label sets")
def test_eval_harness(ctx):
    run = subprocess.run([sys.executable, "scripts/evaluate.py", "--only", "tl,lane"],
                         cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-800:]
    tl = subprocess.run([sys.executable, "scripts/eval_traffic_lights.py"],
                        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert tl.returncode == 0, tl.stderr[-800:]
    det = subprocess.run([sys.executable, "scripts/eval_detection.py", "--models", "yolo11n.pt"],
                         cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert det.returncode == 0, det.stderr[-800:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\downloads\Wyamo Car video.mp4")
    args = ap.parse_args()
    ctx = {"video": args.video, "tmp": Path(tempfile.mkdtemp(prefix="perception_reg_"))}

    for fn in (test_paths, test_lane_profiles, test_tl_reject, test_images,
               test_missing_image, test_video, test_cli, test_eval_harness):
        fn(ctx)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    if passed != len(RESULTS):
        for name, ok, err in RESULTS:
            if not ok:
                print(f"  FAILED: {name} -> {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
