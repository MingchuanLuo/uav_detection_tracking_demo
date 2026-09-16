"""Shared video-inference helpers for Phase E detection and tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from video_utils import VideoMetadata


def validate_probability(value: float, name: str, *, allow_zero: bool = True) -> None:
    """Validate a command-line probability or overlap threshold."""

    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value > 1.0:
        bracket = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {bracket}; received {value}.")


def prepare_outputs(paths: Iterable[Path], overwrite: bool) -> list[Path]:
    """Resolve output paths, protect existing files, and create parent folders."""

    resolved = [path.expanduser().resolve() for path in paths]
    existing = [path for path in resolved if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to replace existing output(s):\n"
            f"{joined}\nPass --overwrite only when replacement is intentional."
        )
    for path in resolved:
        path.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def create_video_writer(
    output_path: Path,
    metadata: VideoMetadata,
    codec: str,
) -> cv2.VideoWriter:
    """Create an OpenCV writer that preserves source geometry and frame rate."""

    if len(codec) != 4 or not codec.isascii():
        raise ValueError("--codec must be a four-character ASCII FOURCC such as mp4v.")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"OpenCV could not create {output_path} with codec {codec!r}."
        )
    return writer


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a stable, human-readable UTF-8 JSON artifact."""

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable, high-contrast BGR color for a tracker ID."""

    palette = (
        (60, 220, 60),
        (255, 170, 30),
        (50, 210, 255),
        (230, 80, 210),
        (255, 90, 70),
        (80, 200, 240),
    )
    return palette[track_id % len(palette)]


def draw_box_label(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw one clipped bounding box with a readable label background."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    thickness = max(2, round(min(width, height) / 540))
    font_scale = max(0.55, min(width, height) / 1500)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_top = y1 - text_height - baseline - 8
    if label_top < 0:
        label_top = y1
    label_bottom = min(height - 1, label_top + text_height + baseline + 8)
    label_right = min(width - 1, x1 + text_width + 10)
    cv2.rectangle(frame, (x1, label_top), (label_right, label_bottom), color, -1)
    text_y = min(label_bottom - baseline - 4, height - baseline - 1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """Draw a translucent status panel in the upper-left corner."""

    if not lines:
        return
    height, width = frame.shape[:2]
    font_scale = max(0.55, min(width, height) / 1600)
    thickness = max(1, round(min(width, height) / 800))
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        for line in lines
    ]
    line_height = max(size[1] for size in sizes) + 12
    panel_width = min(width, max(size[0] for size in sizes) + 24)
    panel_height = min(height, line_height * len(lines) + 12)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 8 + line_height * (index + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )


def draw_trajectory(
    frame: np.ndarray,
    points: Iterable[tuple[int, int]],
    color: tuple[int, int, int],
) -> None:
    """Draw a recent center-point trajectory when at least two points exist."""

    point_list = list(points)
    if len(point_list) < 2:
        return
    trajectory = np.asarray(point_list, dtype=np.int32).reshape((-1, 1, 2))
    thickness = max(2, round(min(frame.shape[:2]) / 540))
    cv2.polylines(frame, [trajectory], False, color, thickness, cv2.LINE_AA)


def image_plane_direction(dx: float, dy: float, stationary_threshold: float) -> str:
    """Map an image displacement to one of eight directions; image y grows downward."""

    if math.hypot(dx, dy) < stationary_threshold:
        return "stationary"
    angle = math.degrees(math.atan2(-dy, dx)) % 360.0
    directions = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
    return directions[int((angle + 22.5) // 45.0) % 8]


def processing_fps(processed_frames: int, started_at: float, now: float) -> float:
    """Return cumulative end-to-end processing throughput."""

    elapsed = now - started_at
    return processed_frames / elapsed if elapsed > 0 else 0.0

