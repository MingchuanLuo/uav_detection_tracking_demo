"""Fine-tune a pretrained Ultralytics YOLO detector on the UAV dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from yolo_runtime import (
    PROJECT_ROOT,
    YOLO,
    environment_summary,
    print_environment,
    select_device,
    validate_dataset_config,
)


DEFAULT_DATA = PROJECT_ROOT / "configs" / "dataset.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "outputs" / "models"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots" / "training"
DEFAULT_PRETRAINED_DIR = DEFAULT_PROJECT / "pretrained"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained YOLO weights for single-class UAV detection."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--run-name", default="yolo11n_uav")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow Ultralytics to reuse an existing run directory.",
    )
    return parser.parse_args()


def load_pretrained_model(model_argument: str) -> tuple[YOLO, Path | str]:
    """Load official weights and retain newly downloaded assets under outputs/models."""

    if not model_argument.lower().endswith(".pt"):
        raise ValueError(
            "--model must be pretrained .pt weights; training from a .yaml architecture "
            "is intentionally disabled."
        )

    supplied_path = Path(model_argument)
    is_bare_name = supplied_path.parent == Path(".")
    cached_path = DEFAULT_PRETRAINED_DIR / supplied_path.name
    if is_bare_name and cached_path.is_file():
        return YOLO(str(cached_path)), cached_path

    existed_before = supplied_path.is_file()
    model = YOLO(model_argument)
    if is_bare_name and not existed_before and supplied_path.is_file():
        DEFAULT_PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(supplied_path), str(cached_path))
        return model, cached_path
    return model, supplied_path.resolve() if supplied_path.is_file() else model_argument


def copy_training_artifacts(save_dir: Path) -> None:
    DEFAULT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = save_dir / "results.csv"
    if results_csv.is_file():
        shutil.copy2(results_csv, DEFAULT_METRICS_DIR / "training_results.csv")

    for pattern in ("*.png", "labels*.jpg", "train_batch*.jpg", "val_batch*.jpg"):
        for source_path in save_dir.glob(pattern):
            if source_path.is_file():
                shutil.copy2(source_path, DEFAULT_PLOTS_DIR / source_path.name)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch == 0:
        raise ValueError("epochs and imgsz must be positive, and batch cannot be zero.")
    if args.workers < 0 or args.patience < 0:
        raise ValueError("workers and patience cannot be negative.")

    data_path = args.data.expanduser().resolve()
    split_paths = validate_dataset_config(data_path)
    selected_device = select_device(args.device)
    environment = environment_summary(selected_device)
    print_environment(environment)
    for split_name, split_path in split_paths.items():
        print(f"{split_name.capitalize():<20}{len(list(split_path.glob('*.jpg')))} images")

    project_dir = args.project.expanduser().resolve()
    planned_run_dir = project_dir / args.run_name
    if planned_run_dir.exists() and not args.exist_ok:
        raise FileExistsError(
            f"Training run already exists: {planned_run_dir}. "
            "Use a new --run-name or explicitly pass --exist-ok."
        )

    model, pretrained_source = load_pretrained_model(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=selected_device,
        workers=args.workers,
        patience=args.patience,
        project=str(project_dir),
        name=args.run_name,
        exist_ok=args.exist_ok,
        pretrained=True,
        seed=args.seed,
        deterministic=True,
        plots=True,
        amp=True,
        cache=args.cache,
    )

    save_dir = Path(model.trainer.save_dir).resolve()
    best_path = Path(model.trainer.best).resolve()
    last_path = Path(model.trainer.last).resolve()
    if not best_path.is_file():
        raise RuntimeError(f"Training finished without best weights: {best_path}")

    DEFAULT_PROJECT.mkdir(parents=True, exist_ok=True)
    stable_best = DEFAULT_PROJECT / "uav_detector_best.pt"
    stable_last = DEFAULT_PROJECT / "uav_detector_last.pt"
    shutil.copy2(best_path, stable_best)
    if last_path.is_file():
        shutil.copy2(last_path, stable_last)
    copy_training_artifacts(save_dir)

    summary = {
        "environment": environment,
        "dataset": str(data_path),
        "pretrained_source": str(pretrained_source),
        "parameters": {
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": selected_device,
            "workers": args.workers,
            "patience": args.patience,
            "seed": args.seed,
        },
        "save_dir": str(save_dir),
        "best_checkpoint": str(stable_best),
        "last_checkpoint": str(stable_last),
    }
    DEFAULT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DEFAULT_METRICS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Training run: {save_dir}")
    print(f"Best model:  {stable_best}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()

