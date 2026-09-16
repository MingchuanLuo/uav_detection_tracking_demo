"""Extract chronologically ordered candidate frames for manual UAV screening."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from video_utils import open_video, read_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "Quadcopter_(drone).webm"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted_frames"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "frame_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract candidate frames without deciding whether a UAV is present."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--every-n-frames",
        type=int,
        help="Save one image for every N decoded source frames.",
    )
    sampling.add_argument(
        "--target-fps",
        type=float,
        help="Approximate extraction rate (default: 3.0 FPS).",
    )

    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only existing frame_*.jpg files and the manifest.",
    )
    return parser.parse_args()


def should_extract(
    frame_index: int,
    every_n_frames: int | None,
    next_target_frame: float,
) -> bool:
    if every_n_frames is not None:
        return frame_index % every_n_frames == 0
    return frame_index + 1e-9 >= next_target_frame


def main() -> None:
    args = parse_args()
    if args.every_n_frames is None and args.target_fps is None:
        args.target_fps = 3.0

    if args.every_n_frames is not None and args.every_n_frames < 1:
        raise ValueError("--every-n-frames must be at least 1.")
    if args.target_fps is not None and args.target_fps <= 0:
        raise ValueError("--target-fps must be greater than 0.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")

    video_path = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(output_dir.glob("frame_*.jpg"))
    existing_outputs = existing_frames or manifest_path.exists()
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Candidate frames or the manifest already exist. Review them, or rerun "
            "with --overwrite only if you intentionally want to regenerate Phase A."
        )
    if args.overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    capture = open_video(video_path)
    manifest_rows: list[dict[str, str | int]] = []
    decoded_frames = 0
    try:
        metadata = read_metadata(capture)
        if args.target_fps is not None and args.target_fps > metadata.fps:
            raise ValueError(
                f"--target-fps ({args.target_fps}) cannot exceed source FPS "
                f"({metadata.fps:.6f})."
            )

        target_interval = (
            metadata.fps / args.target_fps if args.target_fps is not None else None
        )
        next_target_frame = 0.0

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frames += 1

            if should_extract(frame_index, args.every_n_frames, next_target_frame):
                candidate_number = len(manifest_rows) + 1
                filename = f"frame_{candidate_number:06d}.jpg"
                output_path = output_dir / filename
                written = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                if not written:
                    raise RuntimeError(f"Failed to write candidate image: {output_path}")

                manifest_rows.append(
                    {
                        "filename": filename,
                        "original_frame_number": frame_index + 1,
                        "timestamp_seconds": f"{frame_index / metadata.fps:.6f}",
                        "selection_status": "unreviewed",
                    }
                )

                if target_interval is not None:
                    next_target_frame += target_interval

            frame_index += 1
    finally:
        capture.release()

    if not manifest_rows:
        raise RuntimeError("No candidate frames were extracted.")

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "filename",
                "original_frame_number",
                "timestamp_seconds",
                "selection_status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    method = (
        f"every {args.every_n_frames} source frames"
        if args.every_n_frames is not None
        else f"target {args.target_fps:g} FPS"
    )
    print(f"Source:            {video_path}")
    print(f"Sampling:          {method}")
    print(f"Decoded frames:    {decoded_frames}")
    print(f"Candidate images:  {len(manifest_rows)}")
    print(f"Candidate folder:  {output_dir}")
    print(f"Manifest:          {manifest_path}")
    print("Selection status:  unreviewed (no automatic UAV filtering was performed)")


if __name__ == "__main__":
    main()
