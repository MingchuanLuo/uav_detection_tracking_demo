"""Inspect the source UAV video and save representative preview frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from video_utils import open_video, read_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "Quadcopter_(drone).webm"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "previews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report video properties and save evenly spaced preview frames."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def evenly_spaced_indices(total_frames: int, count: int) -> list[int]:
    """Return unique, inclusive frame indices spread over the full video."""

    if count < 1:
        raise ValueError("--preview-count must be at least 1.")
    count = min(count, total_frames)
    if count == 1:
        return [0]
    return sorted(
        {round(index * (total_frames - 1) / (count - 1)) for index in range(count)}
    )


def main() -> None:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")

    video_path = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = open_video(video_path)
    last_frame = None
    last_frame_index = -1
    try:
        metadata = read_metadata(capture)
        target_indices = evenly_spaced_indices(metadata.total_frames, args.preview_count)
        target_set = set(target_indices)
        saved_previews: list[dict[str, int | float | str]] = []

        frame_index = 0
        while target_set:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame = frame
            last_frame_index = frame_index

            if frame_index in target_set:
                timestamp_seconds = frame_index / metadata.fps
                filename = (
                    f"preview_{len(saved_previews) + 1:02d}_"
                    f"frame_{frame_index + 1:06d}_"
                    f"time_{timestamp_seconds:07.3f}s.jpg"
                )
                output_path = output_dir / filename
                written = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                if not written:
                    raise RuntimeError(f"Failed to write preview image: {output_path}")

                saved_previews.append(
                    {
                        "filename": filename,
                        "original_frame_number": frame_index + 1,
                        "timestamp_seconds": round(timestamp_seconds, 6),
                    }
                )
                target_set.remove(frame_index)

            frame_index += 1
    finally:
        capture.release()

    # Some WebM containers report one more indexed frame than the decoder can
    # return. Preserve that fact in the report and still provide an end preview.
    if (
        len(target_set) == 1
        and last_frame is not None
        and last_frame_index not in target_indices
        and next(iter(target_set)) > last_frame_index
    ):
        timestamp_seconds = last_frame_index / metadata.fps
        filename = (
            f"preview_{len(saved_previews) + 1:02d}_"
            f"frame_{last_frame_index + 1:06d}_"
            f"time_{timestamp_seconds:07.3f}s.jpg"
        )
        output_path = output_dir / filename
        written = cv2.imwrite(
            str(output_path),
            last_frame,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
        )
        if not written:
            raise RuntimeError(f"Failed to write preview image: {output_path}")
        saved_previews.append(
            {
                "filename": filename,
                "original_frame_number": last_frame_index + 1,
                "timestamp_seconds": round(timestamp_seconds, 6),
            }
        )
        target_set.clear()

    if target_set:
        missing = ", ".join(str(index + 1) for index in sorted(target_set))
        raise RuntimeError(f"The decoder ended before preview frames were read: {missing}")

    report = {
        "source_video": str(video_path),
        **metadata.to_dict(),
        "successfully_decoded_frames": frame_index,
        "preview_count": len(saved_previews),
        "previews": saved_previews,
    }
    report_path = output_dir / "video_metadata.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Source:       {video_path}")
    print(f"Resolution:   {metadata.width} x {metadata.height}")
    print(f"FPS:          {metadata.fps:.6f}")
    print(f"Total frames: {metadata.total_frames}")
    print(f"Decoded:      {frame_index} frames")
    print(f"Duration:     {metadata.duration_seconds:.3f} seconds")
    print(f"Codec:        {metadata.codec}")
    print(f"Previews:     {len(saved_previews)} saved to {output_dir}")
    print(f"Metadata:     {report_path}")


if __name__ == "__main__":
    main()
