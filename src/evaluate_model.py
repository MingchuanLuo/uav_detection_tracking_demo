"""Evaluate the best UAV detector and export stable metric files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from yolo_runtime import (
    PROJECT_ROOT,
    YOLO,
    environment_summary,
    print_environment,
    select_device,
    validate_dataset_config,
)


DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "uav_detector_best.pt"
DEFAULT_DATA = PROJECT_ROOT / "configs" / "dataset.yaml"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained UAV detector on temporal validation/test splits."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test", "both"), default="both")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow replacement of existing evaluation plot directories.",
    )
    return parser.parse_args()


def metric_row(metrics: object, split: str, image_count: int) -> dict[str, object]:
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    speed = getattr(metrics, "speed", {}) or {}
    return {
        "split": split,
        "images": image_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "preprocess_ms_per_image": float(speed.get("preprocess", 0.0)),
        "inference_ms_per_image": float(speed.get("inference", 0.0)),
        "loss_ms_per_image": float(speed.get("loss", 0.0)),
        "postprocess_ms_per_image": float(speed.get("postprocess", 0.0)),
    }


def main() -> None:
    args = parse_args()
    if args.imgsz < 32 or args.batch == 0 or args.workers < 0:
        raise ValueError("imgsz must be positive, batch nonzero, and workers nonnegative.")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("conf must be in [0,1] and iou in (0,1].")

    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    split_paths = validate_dataset_config(data_path)
    selected_device = select_device(args.device)
    environment = environment_summary(selected_device)
    print_environment(environment)

    splits = ("val", "test") if args.split == "both" else (args.split,)
    for split in splits:
        output_dir = DEFAULT_PLOTS_DIR / f"evaluation_{split}"
        if output_dir.exists() and not args.exist_ok:
            raise FileExistsError(
                f"Evaluation output already exists: {output_dir}. "
                "Pass --exist-ok only when replacement is intentional."
            )

    model = YOLO(str(model_path))
    rows: list[dict[str, object]] = []
    for split in splits:
        metrics = model.val(
            data=str(data_path),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=selected_device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            project=str(DEFAULT_PLOTS_DIR),
            name=f"evaluation_{split}",
            exist_ok=args.exist_ok,
            plots=True,
            verbose=True,
        )
        rows.append(metric_row(metrics, split, len(list(split_paths[split].glob("*.jpg")))))

    DEFAULT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DEFAULT_METRICS_DIR / "evaluation_metrics.csv"
    json_path = DEFAULT_METRICS_DIR / "evaluation_metrics.json"
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "environment": environment,
                "model": str(model_path),
                "dataset": str(data_path),
                "imgsz": args.imgsz,
                "batch": args.batch,
                "confidence_threshold": args.conf,
                "iou_threshold": args.iou,
                "metrics": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for row in rows:
        print(
            f"{row['split']}: P={row['precision']:.4f}, R={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, mAP50={row['map50']:.4f}, "
            f"mAP50-95={row['map50_95']:.4f}"
        )
    print(f"Metrics CSV:  {csv_path}")
    print(f"Metrics JSON: {json_path}")


if __name__ == "__main__":
    main()

