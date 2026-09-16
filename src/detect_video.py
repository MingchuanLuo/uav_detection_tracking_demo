"""Run the trained UAV detector over the complete source video."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from inference_utils import (
    create_video_writer,
    draw_box_label,
    draw_hud,
    prepare_outputs,
    processing_fps,
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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "detections" / "Quadcopter_detection.mp4"
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "detections" / "detections.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "detections" / "detection_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect UAVs in the complete source video and save annotated MP4/CSV output."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Detection confidence threshold; 0.30 removes observed borderline false positives.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="NMS IoU threshold; 0.50 suppresses duplicate boxes on the single UAV.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=1,
        help="Maximum detections per frame; this source video contains one UAV.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--codec", default="mp4v")
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
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
    )
    frames_processed = 0
    frames_with_detections = 0
    detection_count = 0
    confidence_sum = 0.0
    confidence_min: float | None = None
    confidence_max: float | None = None
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
                result = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=selected_device,
                    classes=[0],
                    max_det=args.max_det,
                    verbose=False,
                )[0]
                frame_detections = 0
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    coordinates = boxes.xyxy.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    class_ids = boxes.cls.int().cpu().tolist()
                    for coordinates_row, confidence, class_id in zip(
                        coordinates, confidences, class_ids, strict=True
                    ):
                        x1, y1, x2, y2 = coordinates_row
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        label = f"drone {confidence:.2f}"
                        draw_box_label(
                            frame,
                            tuple(round(value) for value in coordinates_row),
                            label,
                            (60, 220, 60),
                        )
                        csv_writer.writerow(
                            {
                                "frame_number": frame_number,
                                "timestamp_seconds": f"{(frame_number - 1) / metadata.fps:.6f}",
                                "class_id": class_id,
                                "class_name": "drone",
                                "confidence": f"{confidence:.6f}",
                                "x1": f"{x1:.3f}",
                                "y1": f"{y1:.3f}",
                                "x2": f"{x2:.3f}",
                                "y2": f"{y2:.3f}",
                                "center_x": f"{center_x:.3f}",
                                "center_y": f"{center_y:.3f}",
                            }
                        )
                        frame_detections += 1
                        detection_count += 1
                        confidence_sum += confidence
                        confidence_min = (
                            confidence
                            if confidence_min is None
                            else min(confidence_min, confidence)
                        )
                        confidence_max = (
                            confidence
                            if confidence_max is None
                            else max(confidence_max, confidence)
                        )

                if frame_detections:
                    frames_with_detections += 1
                frames_processed = frame_number
                now = time.perf_counter()
                current_fps = processing_fps(frames_processed, started_at, now)
                draw_hud(
                    frame,
                    [
                        f"UAV detection | frame {frame_number}",
                        f"detections: {frame_detections}",
                        f"processing: {current_fps:.1f} FPS | source: {metadata.fps:.1f} FPS",
                    ],
                )
                writer.write(frame)
                if frame_number % 100 == 0:
                    print(
                        f"Processed {frame_number}/{metadata.total_frames} frames "
                        f"({current_fps:.1f} FPS)"
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
        "detections_csv": str(csv_path),
        "scope_note": (
            "This source video contains one UAV; max_det=1 suppresses duplicate "
            "whole-object and part detections."
        ),
        "source_metadata": metadata.to_dict(),
        "parameters": {
            "imgsz": args.imgsz,
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "max_detections_per_frame": args.max_det,
            "device": selected_device,
            "codec": args.codec,
            "max_frames": args.max_frames,
        },
        "frames_processed": frames_processed,
        "frames_with_detections": frames_with_detections,
        "detection_count": detection_count,
        "mean_confidence": confidence_sum / detection_count if detection_count else None,
        "min_confidence": confidence_min,
        "max_confidence": confidence_max,
        "elapsed_seconds": elapsed,
        "end_to_end_processing_fps": frames_processed / elapsed,
    }
    write_json(summary_path, summary)
    print(f"Detection video: {output_path}")
    print(f"Detections CSV:  {csv_path}")
    print(f"Summary:         {summary_path}")
    print(
        f"Frames={frames_processed}, detection_frames={frames_with_detections}, "
        f"detections={detection_count}, processing={frames_processed / elapsed:.1f} FPS"
    )


if __name__ == "__main__":
    main()
