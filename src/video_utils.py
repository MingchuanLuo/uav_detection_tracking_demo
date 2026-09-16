"""Shared helpers for opening videos and reading OpenCV metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    """Basic properties reported by the video container/decoder."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def decode_fourcc(value: float) -> str:
    """Turn OpenCV's numeric FOURCC value into a readable codec identifier."""

    integer_value = int(value)
    if integer_value <= 0:
        return "unknown"

    characters = [chr((integer_value >> (8 * index)) & 0xFF) for index in range(4)]
    codec = "".join(character for character in characters if character.isprintable())
    return codec.strip() or "unknown"


def open_video(video_path: Path) -> cv2.VideoCapture:
    """Open a video and raise a useful error when OpenCV cannot decode it."""

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open the video: {video_path}")
    return capture


def read_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    """Read video properties from an already-open OpenCV capture."""

    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    duration_seconds = total_frames / fps if fps > 0 else 0.0
    codec = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))

    if width <= 0 or height <= 0:
        raise RuntimeError("OpenCV reported an invalid video resolution.")
    if fps <= 0:
        raise RuntimeError("OpenCV could not determine a valid source FPS.")
    if total_frames <= 0:
        raise RuntimeError("OpenCV could not determine the total frame count.")

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration_seconds,
        codec=codec,
    )

