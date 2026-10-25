"""Run the full pipeline on the local images and one clip, per lane model.

    python scripts/run_integrated.py --tag integrated_v17
    python scripts/run_integrated.py --tag integrated_v17 --models tusimple --frames 30

Writes to outputs/runs/<tag>/:

  <image>_<model>.jpg        full pipeline, every module
  <clip>_<model>.mp4         annotated video
  lane_models_compare.jpg    the same 3 frames under each lane checkpoint
  run.json                   per-image telemetry, including why a corridor was
                             withheld, plus measured timings

No metrics are invented here: the JSON carries detection counts, latencies and
corridor diagnostics exactly as the pipeline reports them. Accuracy lives in
scripts/eval_corridor.py and scripts/eval_lane.py, which score against labels.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def images():
    out = []
    for name in ("sir_image.jpg", "sir_image2.jpg", "sir_image3.jpg",
                 "sir_image7.jpeg", "sir_image10.jpeg"):
        p = ROOT / "my_test_images" / name
        if p.exists():
            out.append(p)
    for name in ("highway_road.jpg", "wyamo_f10.jpg", "wyamo_f200.jpg"):
        p = config.TEST_DATA_DIR / name
        if p.exists():
            out.append(p)
    return out


def clip():
    vids = sorted((config.TEST_DATA_DIR / "lane_clips").glob("F_*.webm"))
    return vids[0] if vids else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="integrated_run")
    ap.add_argument("--models", nargs="+", default=["tusimple", "culane"])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    from main import PerceptionPipeline

    report = {"tag": args.tag, "device": config.DEVICE, "models": {}, "images": []}
    tiles = []
    vid = clip()

    for model in args.models:
        print(f"\n=== lane model: {model} ===")
        t0 = time.perf_counter()
        # pass the profile explicitly: setting config.LANE_MODEL after importing
        # main used to be ignored, so both "models" ran tusimple/stretch
        pipeline = PerceptionPipeline(device=config.DEVICE, lane_model=model,
                                      lane_preprocess="stretch")
        assert pipeline.lane_detector.model_key == model, pipeline.lane_detector.model_key
        init_s = time.perf_counter() - t0
        entry = {"init_seconds": round(init_s, 2), "per_image": {}, "video": None}

        for img in images():
            # independent stills: do not let one scene's lane/temporal state bleed
            # into the next (this was a real defect in the lane evaluator).
            if pipeline.lane_detector is not None:
                pipeline.lane_detector.reset()
            if pipeline.tl_classifier is not None:
                pipeline.tl_classifier.reset_temporal()
            _, telem = pipeline.process_image(
                image_input=str(img),
                output_image_path=str(out_dir / f"{img.stem}_{model}.jpg"),
                modules="all",
            )
            mc = telem["module_details"]["module_C"]
            entry["per_image"][img.stem] = {
                "latency_ms": telem["latency_ms"],
                "vehicles": telem["detection_counts"]["vehicles"],
                "traffic_lights": telem["detection_counts"]["traffic_lights"],
                "tl_states": [t["classified_state"] for t in
                              telem["module_details"]["module_B"]["traffic_lights"]],
                "lanes": telem["detection_counts"]["active_lanes"],
                "lane_point_counts": mc["lane_point_counts"],
                "corridor": mc["corridor"],
            }
            c = mc["corridor"]
            print(f"  {img.stem:26s} {telem['latency_ms']:7.1f}ms  "
                  f"lanes {telem['detection_counts']['active_lanes']}  "
                  f"corridor {'YES' if c['established'] else 'no'}"
                  f"{'' if c['established'] else ' - ' + str(c['reason'])}")

        if vid is not None and not args.skip_video:
            summary = pipeline.process_video(
                video_path=str(vid),
                output_video_path=str(out_dir / f"{vid.stem.split('_')[0]}_{model}.mp4"),
                max_frames=args.frames,
                save_keyframes=False,
            )
            entry["video"] = {k: summary[k] for k in
                              ("frames_processed", "average_fps", "wall_clock_time_seconds")}
            cap = cv2.VideoCapture(str(out_dir / f"{vid.stem.split('_')[0]}_{model}.mp4"))
            for idx in (20, 40, 55):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, f = cap.read()
                if ok:
                    tiles.append((f"{model} #{idx}", f))
            cap.release()
        report["models"][model] = entry

    if tiles:
        h = 420
        row = []
        for label, img in tiles:
            s = h / img.shape[0]
            t = cv2.resize(img, (int(img.shape[1] * s), h))
            cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            row.append(t)
        cv2.imwrite(str(out_dir / "lane_models_compare.jpg"), np.hstack(row))
        print(f"\nmontage: {out_dir / 'lane_models_compare.jpg'}")

    (out_dir / "run.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"telemetry: {out_dir / 'run.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
