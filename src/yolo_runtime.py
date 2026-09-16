"""Shared runtime helpers for local Ultralytics training and evaluation."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_CONFIG_ROOT = PROJECT_ROOT / ".ultralytics_config"
ULTRALYTICS_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_ROOT))

import torch  # noqa: E402
import ultralytics  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402


def configure_ultralytics() -> None:
    """Keep Ultralytics state inside the project and disable cloud syncing."""

    settings.update(
        {
            "datasets_dir": str(PROJECT_ROOT / "data"),
            "weights_dir": str(PROJECT_ROOT / "outputs" / "models" / "pretrained"),
            "runs_dir": str(PROJECT_ROOT / "outputs"),
            "sync": False,
            "clearml": False,
            "comet": False,
            "dvc": False,
            "mlflow": False,
            "raytune": False,
            "tensorboard": False,
            "wandb": False,
            "vscode_msg": False,
            "openvino_msg": False,
        }
    )


def select_device(requested: str) -> str:
    """Resolve 'auto' to CUDA 0 when available, otherwise CPU."""

    normalized = requested.strip().lower()
    selected = "0" if normalized == "auto" and torch.cuda.is_available() else normalized
    if normalized == "auto" and not torch.cuda.is_available():
        selected = "cpu"

    wants_cuda = selected != "cpu"
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{requested}' requests CUDA, but torch.cuda.is_available() is False."
        )
    return selected


def environment_summary(selected_device: str) -> dict[str, object]:
    """Return the versions and accelerator details required for reproducibility."""

    cuda_available = torch.cuda.is_available()
    summary: dict[str, object] = {
        "python_torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": cuda_available,
        "pytorch_cuda_version": torch.version.cuda,
        "selected_device": selected_device,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_model": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        summary["gpu_memory_gib"] = round(properties.total_memory / 1024**3, 3)
        summary["gpu_compute_capability"] = ".".join(
            str(value) for value in torch.cuda.get_device_capability(0)
        )
    return summary


def print_environment(summary: dict[str, object]) -> None:
    print(f"PyTorch version:       {summary['python_torch_version']}")
    print(f"Ultralytics version:   {summary['ultralytics_version']}")
    print(f"CUDA available:        {summary['cuda_available']}")
    print(f"PyTorch CUDA version:  {summary['pytorch_cuda_version']}")
    print(f"GPU model:             {summary['gpu_model']}")
    print(f"Selected device:       {summary['selected_device']}")


def validate_dataset_config(config_path: Path) -> dict[str, Path]:
    """Resolve configured split paths and verify one-to-one image/label pairs."""

    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset config is not a mapping: {config_path}")
    if data.get("names") != {0: "drone"}:
        raise ValueError("Dataset config must contain the single class mapping 0: drone.")

    root_value = data.get("path")
    root = config_path.parent
    if root_value:
        configured_root = Path(root_value)
        root = configured_root if configured_root.is_absolute() else root / configured_root

    resolved_splits: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        if split_name not in data:
            raise ValueError(f"Dataset config is missing '{split_name}'.")
        configured_path = Path(data[split_name])
        images_dir = (
            configured_path if configured_path.is_absolute() else root / configured_path
        ).resolve()
        if not images_dir.is_dir():
            raise FileNotFoundError(f"{split_name} image directory not found: {images_dir}")

        parts = list(images_dir.parts)
        try:
            images_index = len(parts) - 1 - parts[::-1].index("images")
        except ValueError as error:
            raise ValueError(f"Split path must contain an images directory: {images_dir}") from error
        parts[images_index] = "labels"
        labels_dir = Path(*parts)
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"{split_name} label directory not found: {labels_dir}")

        image_stems = {path.stem for path in images_dir.glob("*.jpg")}
        label_stems = {path.stem for path in labels_dir.glob("*.txt")}
        if not image_stems or image_stems != label_stems:
            raise ValueError(
                f"{split_name} image/label mismatch: "
                f"images={len(image_stems)}, labels={len(label_stems)}"
            )
        resolved_splits[split_name] = images_dir
    return resolved_splits


configure_ultralytics()

