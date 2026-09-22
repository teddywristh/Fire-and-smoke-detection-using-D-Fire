"""Dataset preparation and loading utilities.

Run this module from the project root to validate raw data, create a clean
processed image tree, and write the shared manifest:

    python -m src.data_loader
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

import config
from src.preprocessing import read_image_metadata, save_processed_image


RAW_FILENAME_RE = re.compile(
    r"^(?P<label>fire|nofire|smoke|smokefire)_(?P<split>train|val|test)_(?P<number>\d+)\.jpg$"
)


@dataclass
class ManifestRow:
    filepath: str
    processed_filepath: str
    split: str
    label: str
    class_index: int
    width: int
    height: int
    mode: str
    image_format: str
    file_size_bytes: int
    raw_sha256: str
    processed_sha256: str
    status: str
    note: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    return os.path.relpath(path.resolve(), config.PROJECT_ROOT).replace(os.sep, "/")


def external_relative(path: Path) -> str:
    return os.path.relpath(path.resolve(), config.PROJECT_ROOT).replace(os.sep, "/")


def expected_processed_path(source_path: Path, split: str, label: str) -> Path:
    return config.PROCESSED_DATA_DIR / split / label / source_path.name


def iter_raw_image_paths() -> list[Path]:
    paths: list[Path] = []
    for split in config.SPLITS:
        for label in config.CLASS_NAMES:
            class_dir = config.DATASET_ROOT / split / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            paths.extend(sorted(class_dir.glob("*.jpg")))
    return paths


def validate_raw_path(path: Path, split: str, label: str) -> list[str]:
    notes: list[str] = []
    match = RAW_FILENAME_RE.match(path.name)
    if not match:
        notes.append("filename_pattern_mismatch")
        return notes

    if match.group("split") != split:
        notes.append("filename_split_mismatch")
    if match.group("label") != label:
        notes.append("filename_label_mismatch")
    return notes


def prepare_dataset() -> list[ManifestRow]:
    if not config.DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {config.DATASET_ROOT}")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[ManifestRow] = []
    raw_hash_counts: Counter[str] = Counter()
    processed_hash_counts: Counter[str] = Counter()

    for source_path in iter_raw_image_paths():
        split = source_path.parent.parent.name
        label = source_path.parent.name
        notes = validate_raw_path(source_path, split, label)

        metadata = read_image_metadata(source_path)
        if metadata.format != "JPEG":
            notes.append("not_pil_jpeg")
        if not metadata.is_jpeg_header:
            notes.append("not_jpeg_header")
        if (metadata.width, metadata.height) != config.IMAGE_SIZE:
            notes.append("raw_size_not_target")

        destination_path = expected_processed_path(source_path, split, label)
        save_processed_image(
            source_path=source_path,
            destination_path=destination_path,
            image_size=config.IMAGE_SIZE,
            resize_policy=config.RESIZE_POLICY,
            jpeg_quality=config.JPEG_QUALITY,
        )

        raw_hash = sha256_file(source_path)
        processed_hash = sha256_file(destination_path)
        raw_hash_counts[raw_hash] += 1
        processed_hash_counts[processed_hash] += 1

        rows.append(
            ManifestRow(
                filepath=external_relative(source_path),
                processed_filepath=project_relative(destination_path),
                split=split,
                label=label,
                class_index=config.CLASS_TO_INDEX[label],
                width=metadata.width,
                height=metadata.height,
                mode=metadata.mode,
                image_format=metadata.format or "",
                file_size_bytes=source_path.stat().st_size,
                raw_sha256=raw_hash,
                processed_sha256=processed_hash,
                status="ok",
                note=";".join(notes) if notes else "clean",
            )
        )

    for row in rows:
        notes = [] if row.note == "clean" else row.note.split(";")
        if raw_hash_counts[row.raw_sha256] > 1:
            notes.append("raw_duplicate_sha256")
        if processed_hash_counts[row.processed_sha256] > 1:
            notes.append("processed_duplicate_sha256")
        row.note = ";".join(notes) if notes else "clean"
        review_markers = ("duplicate", "mismatch", "not_pil_jpeg", "not_jpeg_header")
        row.status = "review" if any(marker in row.note for marker in review_markers) else "ok"

    write_manifest(rows)
    write_report(rows)
    write_sample_grids(rows)
    return rows


def write_manifest(rows: list[ManifestRow]) -> None:
    config.SPLIT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else [field.name for field in ManifestRow.__dataclass_fields__.values()]
    with config.SPLIT_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_manifest(path: Path = config.SPLIT_CSV_PATH) -> list[dict[str, str]]:
    """Load the shared CSV manifest for notebooks or training scripts."""
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_split_rows(split: str, path: Path = config.SPLIT_CSV_PATH) -> list[dict[str, str]]:
    """Return manifest rows for one split."""
    if split not in config.SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return [row for row in load_manifest(path) if row["split"] == split]


def split_class_counts(rows: list[ManifestRow]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        split: {label: 0 for label in config.CLASS_NAMES} for split in config.SPLITS
    }
    for row in rows:
        counts[row.split][row.label] += 1
    return counts


def duplicate_count(rows: list[ManifestRow], field_name: str) -> int:
    counts = Counter(getattr(row, field_name) for row in rows)
    return sum(count for count in counts.values() if count > 1)


def dimension_summary(rows: list[ManifestRow]) -> dict[str, int]:
    counts = Counter(f"{row.width}x{row.height}" for row in rows)
    return dict(counts.most_common())


def write_report(rows: list[ManifestRow]) -> None:
    counts = split_class_counts(rows)
    notes = Counter(row.note for row in rows)
    status_counts = Counter(row.status for row in rows)
    exact_target_count = sum(
        1 for row in rows if (row.width, row.height) == config.IMAGE_SIZE
    )

    summary = {
        "dataset_root": external_relative(config.DATASET_ROOT),
        "processed_data_dir": project_relative(config.PROCESSED_DATA_DIR),
        "manifest": project_relative(config.SPLIT_CSV_PATH),
        "image_size": config.IMAGE_SIZE,
        "resize_policy": config.RESIZE_POLICY,
        "total_images": len(rows),
        "split_class_counts": counts,
        "status_counts": dict(status_counts),
        "raw_exact_target_size_count": exact_target_count,
        "raw_non_target_size_count": len(rows) - exact_target_count,
        "raw_duplicate_files_by_sha256": duplicate_count(rows, "raw_sha256"),
        "processed_duplicate_files_by_sha256": duplicate_count(rows, "processed_sha256"),
        "top_raw_dimensions": dimension_summary(rows),
        "notes": dict(notes),
    }

    with config.PROCESSING_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    lines = [
        "# Data processing report",
        "",
        "This report is generated by `python -m src.data_loader`.",
        "",
        "## Decisions",
        "",
        f"- Raw dataset root: `{external_relative(config.DATASET_ROOT)}`",
        f"- Processed dataset root: `{project_relative(config.PROCESSED_DATA_DIR)}`",
        f"- Manifest: `{project_relative(config.SPLIT_CSV_PATH)}`",
        f"- Target image size: `{config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]}`",
        f"- Resize policy: `{config.RESIZE_POLICY}`",
        "- Original train/val/test split is preserved.",
        "- `Forest Fire_Tester` is excluded from training and final evaluation.",
        "- No augmentation is saved to disk. Augmentation should be train-only during model training.",
        "",
        "## Split/class counts",
        "",
        "| Split | fire | nofire | smoke | smokefire | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for split in config.SPLITS:
        total = sum(counts[split].values())
        lines.append(
            f"| {split} | {counts[split]['fire']} | {counts[split]['nofire']} | "
            f"{counts[split]['smoke']} | {counts[split]['smokefire']} | {total} |"
        )

    lines.extend(
        [
            "",
            "## Quality checks",
            "",
            f"- Total raw images included: {len(rows)}",
            f"- Raw images already at target size: {exact_target_count}",
            f"- Raw images resized to target size: {len(rows) - exact_target_count}",
            f"- Exact raw duplicate files by SHA256: {summary['raw_duplicate_files_by_sha256']}",
            f"- Exact processed duplicate files by SHA256: {summary['processed_duplicate_files_by_sha256']}",
            f"- Status counts: `{dict(status_counts)}`",
            "",
            "## Notes",
            "",
            "- The raw dataset has class-correlated image dimensions.",
            "- Direct resize is used to avoid adding padding borders that could become a shortcut feature.",
            "- Validation and test images are processed independently and receive no augmentation.",
            "- The processed tree mirrors the original split/class structure.",
            "",
            "## Generated files",
            "",
            "- `data/split.csv`",
            "- `data/processed/processing_summary.json`",
            "- `data/processed/processing_report.md`",
            "- `data/processed/samples/train_grid.jpg`",
            "- `data/processed/samples/val_grid.jpg`",
            "- `data/processed/samples/test_grid.jpg`",
        ]
    )

    config.PROCESSING_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sample_grids(rows: list[ManifestRow], samples_per_class: int = 4) -> None:
    sample_dir = config.PROCESSED_DATA_DIR / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    by_split_class: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_split_class[(row.split, row.label)].append(row)

    tile_width, tile_height = config.IMAGE_SIZE
    label_height = 26
    padding = 8

    for split in config.SPLITS:
        grid_width = samples_per_class * tile_width + (samples_per_class + 1) * padding
        grid_height = len(config.CLASS_NAMES) * (tile_height + label_height) + (len(config.CLASS_NAMES) + 1) * padding
        grid = Image.new("RGB", (grid_width, grid_height), "white")
        draw = ImageDraw.Draw(grid)

        for class_index, label in enumerate(config.CLASS_NAMES):
            y = padding + class_index * (tile_height + label_height + padding)
            draw.text((padding, y + 4), f"{split} / {label}", fill=(20, 20, 20))
            for sample_index, row in enumerate(by_split_class[(split, label)][:samples_per_class]):
                x = padding + sample_index * (tile_width + padding)
                with Image.open(config.PROJECT_ROOT / row.processed_filepath) as image:
                    grid.paste(image.convert("RGB"), (x, y + label_height))

        grid.save(sample_dir / f"{split}_grid.jpg", format="JPEG", quality=92)


def console_safe(value: object) -> str:
    """Return text that can be printed on narrow Windows console encodings."""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def main() -> None:
    rows = prepare_dataset()
    print(f"Prepared {len(rows)} images.")
    print(f"Manifest: {console_safe(config.SPLIT_CSV_PATH)}")
    print(f"Report: {console_safe(config.PROCESSING_REPORT_PATH)}")


if __name__ == "__main__":
    main()
