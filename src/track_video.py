"""Track UAV detections with ByteTrack and derive image-plane motion."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from inference_utils import (
    create_video_writer,
    draw_box_label,
    draw_hud,
    draw_trajectory,
    image_plane_direction,
    prepare_outputs,
    processing_fps,
    track_color,
    validate_probability,
    write_json,
)
from video_utils import open_video, read_metadata
from yolo_runtime import (
    PROJECT_ROOT,
    YOLO,
    environment_summary,
    print_environment,
    select_device,
)


DEFAULT_VIDEO = PROJECT_ROOT / "video" / "Quadcopter_(drone).webm"
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "uav_detector_best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "tracking" / "Quadcopter_tracking.mp4"
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "tracking" / "trajectory.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "tracking" / "tracking_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track UAVs with ByteTrack and save trajectories/image-plane motion."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Low detector threshold lets ByteTrack use its second association stage.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="NMS IoU threshold; 0.50 suppresses duplicate boxes before association.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=1,
        help="Maximum detections per frame; increase this for a future multi-UAV model.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--stationary-threshold", type=float, default=1.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional deterministic prefix length for smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_probability(args.conf, "--conf")
    validate_probability(args.iou, "--iou", allow_zero=False)
    if args.imgsz < 32:
        raise ValueError("--imgsz must be at least 32.")
    if args.max_det < 1:
        raise ValueError("--max-det must be at least 1.")
    if args.trail_length < 2:
        raise ValueError("--trail-length must be at least 2.")
    if args.stationary_threshold < 0:
        raise ValueError("--stationary-threshold cannot be negative.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive when provided.")

    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    output_path, csv_path, summary_path = prepare_outputs(
        (args.output, args.csv, args.summary), args.overwrite
    )
    if video_path in (output_path, csv_path, summary_path):
        raise ValueError("An output path cannot replace the source video.")

    selected_device = select_device(args.device)
    environment = environment_summary(selected_device)
    print_environment(environment)
    model = YOLO(str(model_path))
    if model.names != {0: "drone"}:
        raise ValueError(f"Expected the single class 0=drone; model has {model.names}.")

    capture = open_video(video_path)
    metadata = read_metadata(capture)
    writer = create_video_writer(output_path, metadata, args.codec)
    fields = (
        "frame_number",
        "timestamp_seconds",
        "track_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "dx",
        "dy",
        "pixel_displacement",
        "elapsed_since_observation_seconds",
        "image_plane_velocity_px_per_second",
        "image_plane_direction",
    )
    histories: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=args.trail_length)
    )
    last_observation: dict[int, tuple[int, float, float]] = {}
    unique_track_ids: set[int] = set()
    frames_processed = 0
    frames_with_tracks = 0
    tracked_observations = 0
    started_at = time.perf_counter()

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
            csv_writer.writeheader()
            while args.max_frames is None or frames_processed < args.max_frames:
                success, frame = capture.read()
                if not success:
                    break
                frame_number = frames_processed + 1
                result = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=selected_device,
                    classes=[0],
                    max_det=args.max_det,
                    verbose=False,
                )[0]
                current_track_count = 0
                boxes = result.boxes
                if (
                    boxes is not None
                    and len(boxes) > 0
                    and boxes.is_track
                    and boxes.id is not None
                ):
                    coordinates = boxes.xyxy.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    track_ids = boxes.id.int().cpu().tolist()
                    for coordinates_row, confidence, track_id in zip(
                        coordinates, confidences, track_ids, strict=True
                    ):
                        x1, y1, x2, y2 = coordinates_row
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        previous = last_observation.get(track_id)
                        if previous is None:
                            dx = dy = displacement = elapsed_observation = velocity = 0.0
                            direction = "initial"
                        else:
                            previous_frame, previous_x, previous_y = previous
                            dx = center_x - previous_x
                            dy = center_y - previous_y
                            displacement = math.hypot(dx, dy)
                            elapsed_observation = (
                                frame_number - previous_frame
                            ) / metadata.fps
                            velocity = (
                                displacement / elapsed_observation
                                if elapsed_observation > 0
                                else 0.0
                            )
                            direction = image_plane_direction(
                                dx, dy, args.stationary_threshold
                            )

                        last_observation[track_id] = (
                            frame_number,
                            center_x,
                            center_y,
                        )
                        histories[track_id].append((round(center_x), round(center_y)))
                        unique_track_ids.add(track_id)
                        current_track_count += 1
                        tracked_observations += 1
                        color = track_color(track_id)
                        draw_trajectory(frame, histories[track_id], color)
                        box = tuple(round(value) for value in coordinates_row)
                        draw_box_label(
                            frame,
                            box,
                            f"Drone #{track_id} {confidence:.2f}",
                            color,
                        )
                        cv2.circle(
                            frame,
                            (round(center_x), round(center_y)),
                            4,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )
                        motion_text = (
                            "image motion: initial"
                            if direction == "initial"
                            else f"image motion: {velocity:.1f} px/s {direction}"
                        )
                        text_y = (
                            round(y2) + 24
                            if round(y2) + 30 < metadata.height
                            else round(y2) - 16
                        )
                        text_y = min(metadata.height - 12, max(22, text_y))
                        cv2.putText(
                            frame,
                            motion_text,
                            (max(0, round(x1)), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (20, 20, 20),
                            4,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            motion_text,
                            (max(0, round(x1)), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                        csv_writer.writerow(
                            {
                                "frame_number": frame_number,
                                "timestamp_seconds": f"{(frame_number - 1) / metadata.fps:.6f}",
                                "track_id": track_id,
                                "confidence": f"{confidence:.6f}",
                                "x1": f"{x1:.3f}",
                                "y1": f"{y1:.3f}",
                                "x2": f"{x2:.3f}",
                                "y2": f"{y2:.3f}",
                                "center_x": f"{center_x:.3f}",
                                "center_y": f"{center_y:.3f}",
                                "dx": f"{dx:.3f}",
                                "dy": f"{dy:.3f}",
                                "pixel_displacement": f"{displacement:.3f}",
                                "elapsed_since_observation_seconds": f"{elapsed_observation:.6f}",
                                "image_plane_velocity_px_per_second": f"{velocity:.3f}",
                                "image_plane_direction": direction,
                            }
                        )

                if current_track_count:
                    frames_with_tracks += 1
                frames_processed = frame_number
                now = time.perf_counter()
                current_fps = processing_fps(frames_processed, started_at, now)
                draw_hud(
                    frame,
                    [
                        f"ByteTrack UAV tracking | frame {frame_number}",
                        f"active tracks: {current_track_count} | IDs seen: {len(unique_track_ids)}",
                        f"processing: {current_fps:.1f} FPS | source: {metadata.fps:.1f} FPS",
                    ],
                )
                writer.write(frame)
                if frame_number % 100 == 0:
                    print(
                        f"Processed {frame_number}/{metadata.total_frames} frames; "
                        f"IDs={len(unique_track_ids)} ({current_fps:.1f} FPS)"
                    )
    finally:
        capture.release()
        writer.release()

    if frames_processed == 0:
        raise RuntimeError("The source video opened but no frames were decoded.")
    elapsed = time.perf_counter() - started_at
    summary: dict[str, object] = {
        "environment": environment,
        "source_video": str(video_path),
        "model": str(model_path),
        "output_video": str(output_path),
        "trajectory_csv": str(csv_path),
        "source_metadata": metadata.to_dict(),
        "parameters": {
            "tracker": args.tracker,
            "imgsz": args.imgsz,
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "max_detections_per_frame": args.max_det,
            "device": selected_device,
            "codec": args.codec,
            "trail_length": args.trail_length,
            "stationary_threshold_pixels": args.stationary_threshold,
            "max_frames": args.max_frames,
        },
        "coordinate_convention": "x increases right; y increases down",
        "motion_units": "pixels and pixels/second in the image plane",
        "tracker_id_note": (
            "Tracker IDs are trajectory fragments, not a physical UAV count; "
            "this source video contains one UAV."
        ),
        "frames_processed": frames_processed,
        "frames_with_tracks": frames_with_tracks,
        "tracked_observations": tracked_observations,
        "unique_track_count": len(unique_track_ids),
        "track_ids": sorted(unique_track_ids),
        "elapsed_seconds": elapsed,
        "end_to_end_processing_fps": frames_processed / elapsed,
    }
    write_json(summary_path, summary)
    print(f"Tracking video: {output_path}")
    print(f"Trajectory CSV: {csv_path}")
    print(f"Summary:        {summary_path}")
    print(
        f"Frames={frames_processed}, track_frames={frames_with_tracks}, "
        f"unique_IDs={len(unique_track_ids)}, processing={frames_processed / elapsed:.1f} FPS"
    )


if __name__ == "__main__":
    main()
