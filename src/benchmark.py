"""Benchmark end-to-end per-frame YOLO prediction on CPU and CUDA."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import platform
import statistics
import time
from pathlib import Path

from inference_utils import prepare_outputs, validate_probability, write_json
from video_utils import open_video, read_metadata
from yolo_runtime import PROJECT_ROOT, YOLO, torch


DEFAULT_VIDEO = PROJECT_ROOT / "video" / "Quadcopter_(drone).webm"
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "uav_detector_best.pt"
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "metrics" / "benchmark.csv"
DEFAULT_JSON = PROJECT_ROOT / "outputs" / "metrics" / "benchmark_details.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the trained UAV detector on representative video frames."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--details", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--devices", choices=("both", "cpu", "cuda"), default="both")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def evenly_spaced_indices(total_frames: int, count: int) -> list[int]:
    """Return deterministic, inclusive indices over all decodable frames."""

    count = min(total_frames, count)
    if count == 1:
        return [0]
    return [round(index * (total_frames - 1) / (count - 1)) for index in range(count)]


def load_representative_frames(
    video_path: Path, sample_count: int
) -> tuple[list[object], list[int], int, dict[str, int | float | str]]:
    """Decode twice to sample exact positions without retaining the entire video."""

    count_capture = open_video(video_path)
    try:
        metadata = read_metadata(count_capture)
        decoded_count = 0
        while True:
            success, _ = count_capture.read()
            if not success:
                break
            decoded_count += 1
    finally:
        count_capture.release()
    if decoded_count < 1:
        raise RuntimeError("The benchmark source contains no decodable frames.")

    indices = evenly_spaced_indices(decoded_count, sample_count)
    target_set = set(indices)
    frames: list[object] = []
    capture = open_video(video_path)
    try:
        frame_index = 0
        while target_set:
            success, frame = capture.read()
            if not success:
                break
            if frame_index in target_set:
                frames.append(frame.copy())
                target_set.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if target_set or len(frames) != len(indices):
        missing = ", ".join(str(value) for value in sorted(target_set))
        raise RuntimeError(f"Could not load representative frame indices: {missing}")
    return frames, indices, decoded_count, metadata.to_dict()


def percentile(values: list[float], fraction: float) -> float:
    """Calculate a linearly interpolated percentile without an extra dependency."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cpu_model_name() -> str:
    """Read a friendly CPU name on Windows, with a portable fallback."""

    if os.name == "nt":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def synchronize(device_kind: str, gpu_device: int) -> None:
    """Wait for asynchronous CUDA work so wall-clock timings are valid."""

    if device_kind == "cuda":
        torch.cuda.synchronize(gpu_device)


def materialize_boxes(result: object) -> None:
    """Bring final box data to host memory, matching downstream video use."""

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        boxes.data.detach().cpu().numpy()


def benchmark_device(
    model_path: Path,
    frames: list[object],
    device_kind: str,
    gpu_device: int,
    warmup: int,
    repetitions: int,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
) -> tuple[dict[str, object], list[float]]:
    """Benchmark one device after model construction and warm-up."""

    ultralytics_device = "cpu" if device_kind == "cpu" else str(gpu_device)
    device_name = cpu_model_name()
    if device_kind == "cuda":
        device_name = torch.cuda.get_device_name(gpu_device)

    model = YOLO(str(model_path))
    if model.names != {0: "drone"}:
        raise ValueError(f"Expected the single class 0=drone; model has {model.names}.")
    predict_arguments = {
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "device": ultralytics_device,
        "classes": [0],
        "verbose": False,
    }
    for index in range(warmup):
        result = model.predict(frames[index % len(frames)], **predict_arguments)[0]
        materialize_boxes(result)
    synchronize(device_kind, gpu_device)
    if device_kind == "cuda":
        torch.cuda.reset_peak_memory_stats(gpu_device)

    latencies_ms: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    for _ in range(repetitions):
        for frame in frames:
            synchronize(device_kind, gpu_device)
            started_at = time.perf_counter()
            result = model.predict(frame, **predict_arguments)[0]
            materialize_boxes(result)
            synchronize(device_kind, gpu_device)
            latencies_ms.append((time.perf_counter() - started_at) * 1000.0)
            speed = result.speed or {}
            preprocess_ms.append(float(speed.get("preprocess", 0.0)))
            inference_ms.append(float(speed.get("inference", 0.0)))
            postprocess_ms.append(float(speed.get("postprocess", 0.0)))

    mean_latency = statistics.mean(latencies_ms)
    row: dict[str, object] = {
        "device": "CPU" if device_kind == "cpu" else "GPU",
        "device_name": device_name,
        "sample_frames": len(frames),
        "repetitions": repetitions,
        "timed_runs": len(latencies_ms),
        "warmup_runs": warmup,
        "imgsz": imgsz,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": statistics.median(latencies_ms),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "latency_stddev_ms": statistics.pstdev(latencies_ms),
        "fps": 1000.0 / mean_latency,
        "mean_preprocess_ms": statistics.mean(preprocess_ms),
        "mean_model_inference_ms": statistics.mean(inference_ms),
        "mean_postprocess_ms": statistics.mean(postprocess_ms),
        "cuda_synchronized": device_kind == "cuda",
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(gpu_device) / 1024**2
            if device_kind == "cuda"
            else None
        ),
    }
    del model
    gc.collect()
    if device_kind == "cuda":
        torch.cuda.empty_cache()
    return row, latencies_ms


def main() -> None:
    args = parse_args()
    validate_probability(args.conf, "--conf")
    validate_probability(args.iou, "--iou", allow_zero=False)
    if args.sample_count < 2:
        raise ValueError("--sample-count must be at least 2.")
    if args.repetitions < 1 or args.warmup < 1:
        raise ValueError("--repetitions and --warmup must be positive.")
    if args.imgsz < 32 or args.max_det < 1:
        raise ValueError("--imgsz must be at least 32 and --max-det at least 1.")

    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    output_path, details_path = prepare_outputs(
        (args.output, args.details), args.overwrite
    )

    requested_devices = (
        ("cpu", "cuda")
        if args.devices == "both"
        else (args.devices,)
    )
    if "cuda" in requested_devices:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmarking was requested but CUDA is unavailable.")
        if args.gpu_device < 0 or args.gpu_device >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu-device {args.gpu_device} is outside the available device range."
            )

    print("Decoding representative frames...")
    frames, frame_indices, decoded_count, source_metadata = load_representative_frames(
        video_path, args.sample_count
    )
    print(
        f"Loaded {len(frames)} evenly spaced frames from {decoded_count} decoded frames."
    )

    rows: list[dict[str, object]] = []
    raw_latencies: dict[str, list[float]] = {}
    for device_kind in requested_devices:
        print(f"Benchmarking {device_kind.upper()} after {args.warmup} warm-up runs...")
        row, latencies = benchmark_device(
            model_path=model_path,
            frames=frames,
            device_kind=device_kind,
            gpu_device=args.gpu_device,
            warmup=args.warmup,
            repetitions=args.repetitions,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
        )
        rows.append(row)
        raw_latencies[str(row["device"])] = latencies
        print(
            f"{row['device']}: mean={row['mean_latency_ms']:.2f} ms, "
            f"p95={row['p95_latency_ms']:.2f} ms, FPS={row['fps']:.2f}"
        )

    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    details: dict[str, object] = {
        "model": str(model_path),
        "video": str(video_path),
        "source_metadata": source_metadata,
        "successfully_decoded_frames": decoded_count,
        "sample_frame_numbers_one_based": [index + 1 for index in frame_indices],
        "parameters": {
            "sample_count": len(frames),
            "repetitions": args.repetitions,
            "warmup_runs": args.warmup,
            "imgsz": args.imgsz,
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "max_detections_per_frame": args.max_det,
        },
        "measurement_scope": (
            "Per-frame batch=1 wall time from an in-memory BGR NumPy frame through "
            "Ultralytics preprocessing, host/device transfer, model inference, NMS, "
            "and materialization of final boxes in CPU memory. Video decoding, drawing, "
            "encoding, model loading, and warm-up are excluded."
        ),
        "cuda_timing_note": (
            "torch.cuda.synchronize(device) is called immediately before and after "
            "every timed CUDA prediction."
        ),
        "hardware": {
            "cpu": cpu_model_name(),
            "logical_processors": os.cpu_count(),
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(args.gpu_device)
                if torch.cuda.is_available()
                else None
            ),
            "pytorch_version": torch.__version__,
            "pytorch_cuda_version": torch.version.cuda,
        },
        "results": rows,
        "raw_latencies_ms": raw_latencies,
    }
    write_json(details_path, details)
    print(f"Benchmark CSV:     {output_path}")
    print(f"Benchmark details: {details_path}")


if __name__ == "__main__":
    main()
