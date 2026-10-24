"""
Video and Visual Overlay Utilities for Perception System Version 1
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path


def get_video_properties(video_path: str) -> Dict[str, float]:
    """Inspects a video file and returns its dimensions, fps, frame count, duration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = (frame_count / fps) if fps > 0 else 0.0

    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_s,
    }


def create_video_writer(
    out_path: str,
    fps: float,
    width: int,
    height: int,
    preferred_codec: str = "mp4v",
) -> cv2.VideoWriter:
    """
    Creates an OpenCV VideoWriter with fallback codecs to ensure compatibility.
    Tries preferred_codec ('mp4v' or 'avc1'), falls back if writer fails to open.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    codecs_to_try = [preferred_codec, "mp4v", "avc1", "MJPG"]
    # De-duplicate preserving order
    codecs_to_try = list(dict.fromkeys(codecs_to_try))

    writer = None
    for codec in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer

    raise RuntimeError(f"Failed to create VideoWriter for {out_path} with any tried codec: {codecs_to_try}")


def draw_perception_hud(
    frame_bgr: np.ndarray,
    stats: Dict[str, object],
) -> np.ndarray:
    """
    Renders a head-up display (HUD) banner with real-time perception telemetry:
    - Frame count & Progress
    - FPS and Pipeline Latency (ms)
    - Active Tracking IDs & Object counts (Vehicles, Pedestrians)
    - Active Traffic Light State
    - Lane Detection Status
    """
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    # Semi-transparent HUD overlay at the top
    hud_height = max(70, int(h * 0.09))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)

    # Accent divider line (electric cyan)
    cv2.line(annotated, (0, hud_height), (w, hud_height), (255, 180, 0), 2)

    # Scale font size based on image width
    scale = max(0.40, min(0.65, w / 1600.0))
    thickness = 1

    # Row 1: System Title & Telemetry
    title = "PERCEPTION SYSTEM v1 (PRETRAINED BASELINE)"
    fps_val = stats.get("fps", 0.0)
    latency_ms = stats.get("latency_ms", 0.0)
    frame_idx = stats.get("frame_idx", 0)
    total_frames = stats.get("total_frames", 0)
    is_image = stats.get("is_image", False)

    if is_image:
        modules_str = stats.get("modules_str", "A, B, C")
        telemetry = f"[{title}]  STATIC IMAGE | Latency: {latency_ms:.1f} ms | Modules: {modules_str}"
    else:
        frame_str = f"F: {frame_idx}/{total_frames}" if total_frames > 0 else f"F: {frame_idx}"
        telemetry = f"[{title}]  {frame_str} | {fps_val:.1f} FPS ({latency_ms:.1f} ms)"

    # Row 2: Object counts, Traffic Light state, Lane detection
    vehicles = stats.get("vehicles_count", 0)
    pedestrians = stats.get("pedestrians_count", 0)
    active_tracks = stats.get("active_tracks", 0)
    tl_state = stats.get("tl_state", "None")
    lane_status = stats.get("lane_status", "Active")

    if is_image:
        track_str = f"Tracks: {active_tracks} (Init ID)" if active_tracks > 0 else "Tracks: 0"
    else:
        track_str = f"Active Tracks: {active_tracks}"

    stats_line = (
        f"Vehicles: {vehicles} | Pedestrians: {pedestrians} | {track_str} | "
        f"Lanes: {lane_status} | "
    )
    tl_text = f"TL State: {str(tl_state).upper()}"

    # Dynamic scale adjustment if image width is narrow
    (t1_w, _), _ = cv2.getTextSize(telemetry, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    (t2_w, _), _ = cv2.getTextSize(stats_line + tl_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    max_line_w = max(t1_w, t2_w)
    if max_line_w > (w - 24) and scale > 0.28:
        scale = max(0.28, scale * (w - 24) / max(1, max_line_w))

    cv2.putText(
        annotated,
        telemetry,
        (12, int(hud_height * 0.42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # TL state color
    tl_colors = {
        "red": (0, 0, 255),
        "yellow": (0, 215, 255),
        "green": (0, 255, 0),
        "unknown": (180, 180, 180),
        "None": (130, 130, 130),
    }
    tl_col = tl_colors.get(str(tl_state), (200, 200, 200))

    cv2.putText(
        annotated,
        stats_line,
        (12, int(hud_height * 0.85)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 230, 255),
        thickness,
        cv2.LINE_AA,
    )

    # Draw TL State indicator at end of line 2
    (line_w, _), _ = cv2.getTextSize(stats_line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        annotated,
        tl_text,
        (12 + line_w, int(hud_height * 0.85)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        tl_col,
        thickness + 1,
        cv2.LINE_AA,
    )

    return annotated


def save_annotated_frame(frame_bgr: np.ndarray, output_path: str):
    """Saves an annotated frame image to disk, creating parent directories."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame_bgr)
