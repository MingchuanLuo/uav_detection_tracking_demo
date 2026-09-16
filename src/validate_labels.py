"""Strictly validate YOLO labels and render deterministic review previews."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "selected_frames"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "data" / "labels"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "outputs" / "previews" / "annotation_check"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "metrics" / "label_validation.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLO detection labels without modifying them."
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--class-name", default="drone")
    parser.add_argument(
        "--expected-boxes-per-image",
        type=int,
        default=1,
        help="Required box count per image; use 0 to disable this check.",
    )
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def issue(filename: str, message: str, line: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"filename": filename, "message": message}
    if line is not None:
        result["line"] = line
    return result


def validate_label(
    label_path: Path,
    expected_class_id: int,
    expected_box_count: int,
) -> tuple[list[tuple[int, float, float, float, float]], list[dict[str, object]]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    problems: list[dict[str, object]] = []
    try:
        text = label_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        return boxes, [issue(label_path.name, f"cannot read label file: {error}")]

    raw_lines = text.splitlines()
    nonempty_lines = [line for line in raw_lines if line.strip()]
    if not nonempty_lines:
        problems.append(issue(label_path.name, "label file contains no bounding box"))

    if expected_box_count > 0 and len(nonempty_lines) != expected_box_count:
        problems.append(
            issue(
                label_path.name,
                f"expected {expected_box_count} box, found {len(nonempty_lines)}",
            )
        )

    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line:
            problems.append(issue(label_path.name, "blank label line", line_number))
            continue

        fields = line.split()
        if len(fields) != 5:
            problems.append(
                issue(
                    label_path.name,
                    f"expected 5 fields, found {len(fields)}",
                    line_number,
                )
            )
            continue

        try:
            class_id = int(fields[0])
        except ValueError:
            problems.append(
                issue(label_path.name, "class ID is not an integer", line_number)
            )
            continue

        try:
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            problems.append(
                issue(label_path.name, "coordinates are not numeric", line_number)
            )
            continue

        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            problems.append(
                issue(label_path.name, "coordinates must be finite", line_number)
            )
            continue

        if class_id != expected_class_id:
            problems.append(
                issue(
                    label_path.name,
                    f"class ID must be {expected_class_id}, found {class_id}",
                    line_number,
                )
            )
        if not 0.0 <= x_center <= 1.0:
            problems.append(
                issue(label_path.name, "x_center is outside [0, 1]", line_number)
            )
        if not 0.0 <= y_center <= 1.0:
            problems.append(
                issue(label_path.name, "y_center is outside [0, 1]", line_number)
            )
        if not 0.0 < width <= 1.0:
            problems.append(
                issue(label_path.name, "width must be in (0, 1]", line_number)
            )
        if not 0.0 < height <= 1.0:
            problems.append(
                issue(label_path.name, "height must be in (0, 1]", line_number)
            )

        left = x_center - width / 2.0
        right = x_center + width / 2.0
        top = y_center - height / 2.0
        bottom = y_center + height / 2.0
        tolerance = 1e-6
        if (
            left < -tolerance
            or top < -tolerance
            or right > 1.0 + tolerance
            or bottom > 1.0 + tolerance
        ):
            problems.append(
                issue(
                    label_path.name,
                    "bounding box extends outside image boundaries",
                    line_number,
                )
            )

        boxes.append((class_id, x_center, y_center, width, height))

    return boxes, problems


def draw_preview(
    image_path: Path,
    boxes: list[tuple[int, float, float, float, float]],
    class_name: str,
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image for preview: {image_path}")
    image_height, image_width = image.shape[:2]

    cv2.rectangle(image, (0, 0), (image_width, 42), (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        image_path.name,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for _, x_center, y_center, width, height in boxes:
        left = max(0, round((x_center - width / 2.0) * image_width))
        right = min(image_width - 1, round((x_center + width / 2.0) * image_width))
        top = max(0, round((y_center - height / 2.0) * image_height))
        bottom = min(
            image_height - 1, round((y_center + height / 2.0) * image_height)
        )
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)
        label_y = max(62, top - 10)
        cv2.putText(
            image,
            class_name,
            (left, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Cannot write annotation preview: {output_path}")


def main() -> None:
    args = parse_args()
    if args.preview_count < 0:
        raise ValueError("--preview-count cannot be negative.")
    if args.expected_boxes_per_image < 0:
        raise ValueError("--expected-boxes-per-image cannot be negative.")

    images_dir = args.images_dir.expanduser().resolve()
    labels_dir = args.labels_dir.expanduser().resolve()
    preview_dir = args.preview_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    label_paths = sorted(labels_dir.glob("*.txt"))
    problems: list[dict[str, object]] = []
    boxes_by_image: dict[str, list[tuple[int, float, float, float, float]]] = {}

    if not image_paths:
        problems.append(issue(str(images_dir), "no supported images found"))

    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}
    for stem in sorted(image_stems - label_stems):
        problems.append(issue(f"{stem}.jpg", "matching label file is missing"))
    for stem in sorted(label_stems - image_stems):
        problems.append(issue(f"{stem}.txt", "matching selected image is missing"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            problems.append(issue(image_path.name, "image cannot be decoded"))
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        boxes, label_problems = validate_label(
            label_path,
            expected_class_id=args.class_id,
            expected_box_count=args.expected_boxes_per_image,
        )
        boxes_by_image[image_path.name] = boxes
        problems.extend(label_problems)

    report: dict[str, object] = {
        "valid": not problems,
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "expected_class_id": args.class_id,
        "expected_boxes_per_image": args.expected_boxes_per_image,
        "issue_count": len(problems),
        "issues": problems,
        "preview_seed": args.seed,
        "previews": [],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    if problems:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Validation FAILED with {len(problems)} issue(s).")
        for problem in problems:
            line_suffix = (
                f" line {problem['line']}" if "line" in problem else ""
            )
            print(f"- {problem['filename']}{line_suffix}: {problem['message']}")
        print(f"Report: {report_path}")
        raise SystemExit(1)

    preview_dir.mkdir(parents=True, exist_ok=True)
    for old_preview in preview_dir.glob("*.jpg"):
        old_preview.unlink()

    sample_count = min(args.preview_count, len(image_paths))
    sampled_paths = sorted(random.Random(args.seed).sample(image_paths, sample_count))
    preview_names: list[str] = []
    for image_path in sampled_paths:
        output_path = preview_dir / image_path.name
        draw_preview(
            image_path,
            boxes_by_image[image_path.name],
            args.class_name,
            output_path,
        )
        preview_names.append(output_path.name)

    report["previews"] = preview_names
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Validation PASSED.")
    print(f"Images:   {len(image_paths)}")
    print(f"Labels:   {len(label_paths)}")
    print(f"Boxes:    {sum(len(boxes) for boxes in boxes_by_image.values())}")
    print(f"Previews: {len(preview_names)} saved to {preview_dir}")
    print(f"Report:   {report_path}")


if __name__ == "__main__":
    main()

