"""Create leakage-aware chronological YOLO train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

from validate_labels import validate_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "selected_frames"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "data" / "labels"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "frame_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a chronological YOLO dataset without random frame mixing."
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated YOLO dataset.",
    )
    return parser.parse_args()


def allocate_split_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    """Use largest remainders so rounded split counts still sum to total."""

    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remainder = total - sum(counts)
    priority = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in priority[:remainder]:
        counts[index] += 1
    return counts


def load_selected_rows(manifest_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get("selection_status") == "selected"]

    required_columns = {
        "filename",
        "original_frame_number",
        "timestamp_seconds",
        "selection_status",
    }
    if fieldnames is None or not required_columns.issubset(fieldnames):
        raise ValueError(
            f"Manifest must contain columns: {', '.join(sorted(required_columns))}"
        )

    rows.sort(
        key=lambda row: (
            float(row["timestamp_seconds"]),
            int(row["original_frame_number"]),
        )
    )
    return fieldnames, rows


def prepare_empty_structure(output_dir: Path, overwrite: bool) -> None:
    generated_paths = [
        output_dir / kind / split
        for kind in ("images", "labels")
        for split in SPLIT_NAMES
    ]
    generated_paths.append(output_dir / "split_manifest.csv")
    existing = [path for path in generated_paths if path.exists()]
    if existing and not overwrite:
        shown = "\n- ".join(str(path) for path in existing)
        raise FileExistsError(
            "Generated dataset paths already exist. Use --overwrite only when "
            f"regeneration is intentional:\n- {shown}"
        )

    if overwrite:
        for path in generated_paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    for kind in ("images", "labels"):
        for split in SPLIT_NAMES:
            (output_dir / kind / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    ratios = (args.train_ratio, args.val_ratio, test_ratio)
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("Train, validation, and test ratios must all be positive.")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0.")

    images_dir = args.images_dir.expanduser().resolve()
    labels_dir = args.labels_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    for required_path in (images_dir, labels_dir):
        if not required_path.is_dir():
            raise FileNotFoundError(f"Required directory not found: {required_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_fields, selected_rows = load_selected_rows(manifest_path)
    if not selected_rows:
        raise ValueError("The manifest contains no selected images.")

    manifest_names = [row["filename"] for row in selected_rows]
    if len(manifest_names) != len(set(manifest_names)):
        raise ValueError("Selected filenames are duplicated in the manifest.")

    image_names = {path.name for path in images_dir.glob("*.jpg")}
    label_names = {path.name for path in labels_dir.glob("*.txt")}
    expected_images = set(manifest_names)
    expected_labels = {f"{Path(name).stem}.txt" for name in manifest_names}
    if image_names != expected_images:
        missing = sorted(expected_images - image_names)
        extra = sorted(image_names - expected_images)
        raise ValueError(f"Selected image mismatch. Missing={missing}, extra={extra}")
    if label_names != expected_labels:
        missing = sorted(expected_labels - label_names)
        extra = sorted(label_names - expected_labels)
        raise ValueError(f"Label mismatch. Missing={missing}, extra={extra}")

    label_problems: list[str] = []
    for label_name in sorted(label_names):
        _, problems = validate_label(
            labels_dir / label_name,
            expected_class_id=0,
            expected_box_count=1,
        )
        label_problems.extend(
            f"{problem['filename']}: {problem['message']}" for problem in problems
        )
    if label_problems:
        raise ValueError("Label validation failed:\n- " + "\n- ".join(label_problems))

    counts = allocate_split_counts(len(selected_rows), ratios)
    if any(count == 0 for count in counts):
        raise ValueError(f"Every temporal split must contain images; counts={counts}")

    split_rows: dict[str, list[dict[str, str]]] = {}
    start = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        end = start + count
        split_rows[split_name] = selected_rows[start:end]
        start = end

    prepare_empty_structure(output_dir, args.overwrite)
    output_manifest_rows: list[dict[str, str]] = []
    for split_name in SPLIT_NAMES:
        for row in split_rows[split_name]:
            image_name = row["filename"]
            label_name = f"{Path(image_name).stem}.txt"
            shutil.copy2(images_dir / image_name, output_dir / "images" / split_name)
            shutil.copy2(labels_dir / label_name, output_dir / "labels" / split_name)
            output_manifest_rows.append({**row, "split": split_name})

    split_manifest_path = output_dir / "split_manifest.csv"
    split_fields = [*manifest_fields, "split"]
    with split_manifest_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=split_fields)
        writer.writeheader()
        writer.writerows(output_manifest_rows)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "# Paths resolve from this YAML file because `path` is omitted.\n"
        "train: ../data/yolo_dataset/images/train\n"
        "val: ../data/yolo_dataset/images/val\n"
        "test: ../data/yolo_dataset/images/test\n"
        "\n"
        "names:\n"
        "  0: drone\n",
        encoding="utf-8",
    )

    print(f"Chronological dataset created from {len(selected_rows)} labelled images.")
    for split_name, ratio in zip(SPLIT_NAMES, ratios, strict=True):
        rows = split_rows[split_name]
        print(
            f"{split_name:>5}: {len(rows):>3} images "
            f"({ratio:.0%} target), "
            f"{float(rows[0]['timestamp_seconds']):.3f}s to "
            f"{float(rows[-1]['timestamp_seconds']):.3f}s"
        )
    print(f"Dataset:  {output_dir}")
    print(f"Manifest: {split_manifest_path}")
    print(f"Config:   {config_path}")


if __name__ == "__main__":
    main()
