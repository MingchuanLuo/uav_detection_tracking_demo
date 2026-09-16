"""Validate manual frame screening and synchronize the frame manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "frame_manifest.csv"
DEFAULT_EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted_frames"
DEFAULT_SELECTED_DIR = PROJECT_ROOT / "data" / "selected_frames"
DEFAULT_REJECTED_DIR = PROJECT_ROOT / "data" / "rejected_frames"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate screening folders and update selection_status in the manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument("--selected-dir", type=Path, default=DEFAULT_SELECTED_DIR)
    parser.add_argument("--rejected-dir", type=Path, default=DEFAULT_REJECTED_DIR)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any manifest image remains in extracted_frames.",
    )
    return parser.parse_args()


def filenames(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Screening directory not found: {directory}")
    return {path.name for path in directory.iterdir() if path.is_file()}


def preview(items: set[str] | list[str], limit: int = 10) -> str:
    ordered = sorted(items)
    suffix = " ..." if len(ordered) > limit else ""
    return ", ".join(ordered[:limit]) + suffix


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        fieldnames = reader.fieldnames
        rows = list(reader)

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

    manifest_names = [row["filename"] for row in rows]
    duplicate_manifest_names = sorted(
        name for name, count in Counter(manifest_names).items() if count > 1
    )
    if duplicate_manifest_names:
        raise ValueError(
            "Duplicate filenames in manifest: " + preview(duplicate_manifest_names)
        )

    folder_names = {
        "unreviewed": filenames(args.extracted_dir.expanduser().resolve()),
        "selected": filenames(args.selected_dir.expanduser().resolve()),
        "rejected": filenames(args.rejected_dir.expanduser().resolve()),
    }

    locations: dict[str, list[str]] = {}
    for status, names in folder_names.items():
        for name in names:
            locations.setdefault(name, []).append(status)

    multiply_placed = {
        name for name, statuses in locations.items() if len(statuses) > 1
    }
    expected = set(manifest_names)
    observed = set(locations)
    missing = expected - observed
    unknown = observed - expected

    problems: list[str] = []
    if multiply_placed:
        problems.append("present in multiple folders: " + preview(multiply_placed))
    if missing:
        problems.append("missing from all screening folders: " + preview(missing))
    if unknown:
        problems.append("not listed in the manifest: " + preview(unknown))
    if args.require_complete and folder_names["unreviewed"]:
        problems.append(
            "still unreviewed: " + preview(folder_names["unreviewed"])
        )
    if problems:
        raise ValueError("Screening validation failed:\n- " + "\n- ".join(problems))

    for row in rows:
        row["selection_status"] = locations[row["filename"]][0]

    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(manifest_path)

    print(f"Manifest rows: {len(rows)}")
    print(f"Selected:      {len(folder_names['selected'])}")
    print(f"Rejected:      {len(folder_names['rejected'])}")
    print(f"Unreviewed:    {len(folder_names['unreviewed'])}")
    print(f"Updated:       {manifest_path}")


if __name__ == "__main__":
    main()

