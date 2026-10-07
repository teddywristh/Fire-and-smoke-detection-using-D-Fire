# -*- coding: utf-8 -*-
"""
  - kiểm tra dữ liệu thô,
  - tạo cây ảnh đã xử lý (processed),
  - ghi file manifest chung.
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
from src.data.preprocessing import read_image_metadata, save_processed_image


RAW_FILENAME_RE = re.compile(
    r"^(?P<label>fire|nofire|smoke|smokefire)_"
    r"(?P<split>train|val|test)_"
    r"(?P<number>\d+)\.jpg$"
)

@dataclass
class ManifestRow:
    """
    Một hàng trong file manifest CSV, mô tả một ảnh.
    """
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

def sha256_file(path):
    """
    path: Path tới file cần tính hash.
    Trả về: chuỗi hex của SHA256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def project_relative(path):
    rel = os.path.relpath(path.resolve(), config.PROJECT_ROOT)
    return rel.replace(os.sep, "/")

def external_relative(path):
    rel = os.path.relpath(path.resolve(), config.PROJECT_ROOT)
    return rel.replace(os.sep, "/")

def expected_processed_path(source_path, split, label):
    """
    source_path: Path tới ảnh thô.
    split: "train", "val", hoặc "test".
    label: "fire", "nofire", "smoke", "smokefire".
    Trả về: Path tới file sẽ được ghi trong thư mục processed.
    """
    return config.PROCESSED_DATA_DIR / split / label / source_path.name

def iter_raw_image_paths():
    """
    Cấu trúc:
    DATASET_ROOT / split / label / *.jpg với split trong config.SPLITS và label trong config.CLASS_NAMES.
    """
    paths = []
    for split in config.SPLITS:
        for label in config.CLASS_NAMES:
            class_dir = config.DATASET_ROOT / split / label
            if not class_dir.exists():
                raise FileNotFoundError(
                    "Thiếu thư mục class: {}".format(str(class_dir))
                )
            for p in sorted(class_dir.glob("*.jpg")):
                paths.append(p)
    return paths

def validate_raw_path(path, split, label):
    """
    path: Path tới file ảnh thô.
    split, label: giá trị mong đợi dựa trên vị trí thư mục.
    Trả về: danh sách các “note” mô tả vấn đề (nếu có).
    """
    notes = []
    match = RAW_FILENAME_RE.match(path.name)
    if not match:
        notes.append("filename_pattern_mismatch")
        return notes
    if match.group("split") != split:
        notes.append("filename_split_mismatch")
    if match.group("label") != label:
        notes.append("filename_label_mismatch")
    return notes


def prepare_dataset():
    """
    Chuẩn bị dataset: kiểm tra dữ liệu thô, tạo cây ảnh đã xử lý, ghi file manifest. Trả về danh sách các ManifestRow.
    """
    if not config.DATASET_ROOT.exists():
        raise FileNotFoundError(
            "Không tìm thấy thư mục gốc dataset: {}".format(
                str(config.DATASET_ROOT)
            )
        )
    if not config.PROCESSED_DATA_DIR.exists():
        config.PROCESSED_DATA_DIR.mkdir(parents=True)
    rows = []
    raw_hash_counts = Counter()
    processed_hash_counts = Counter()

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
        row = ManifestRow(
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
        rows.append(row)
    for row in rows:
        notes = []
        if row.note == "clean":
            notes = []
        else:
            notes = row.note.split(";")
        if raw_hash_counts[row.raw_sha256] > 1:
            notes.append("raw_duplicate_sha256")
        if processed_hash_counts[row.processed_sha256] > 1:
            notes.append("processed_duplicate_sha256")
        row.note = ";".join(notes) if notes else "clean"
        review_markers = (
            "duplicate",
            "mismatch",
            "not_pil_jpeg",
            "not_jpeg_header",
        )
        need_review = False
        for marker in review_markers:
            if marker in row.note:
                need_review = True
                break

        row.status = "review" if need_review else "ok"
    write_manifest(rows)
    write_report(rows)
    write_sample_grids(rows)
    return rows

def write_manifest(rows):
    """
    Ghi danh sách ManifestRow ra file CSV (split.csv).
    """
    if not config.SPLIT_CSV_PATH.parent.exists():
        config.SPLIT_CSV_PATH.parent.mkdir(parents=True)
    if rows:
        fieldnames = list(asdict(rows[0]).keys())
    else:
        fieldnames = [f.name for f in ManifestRow.__dataclass_fields__.values()]

    with config.SPLIT_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

def load_manifest(path=None):
    """
    Đọc file manifest CSV và trả về danh sách dict (mỗi dict là một hàng).

    path: đường dẫn tới file CSV, mặc định là config.SPLIT_CSV_PATH.
    """
    if path is None:
        path = config.SPLIT_CSV_PATH

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append(row)
        return rows

def get_split_rows(split, path=None):
    """
    split: "train", "val", hoặc "test".
    path: đường dẫn tới file manifest CSV.
    """
    if path is None:
        path = config.SPLIT_CSV_PATH

    if split not in config.SPLITS:
        raise ValueError("Split không hợp lệ: {}".format(split))

    all_rows = load_manifest(path)
    result = []
    for row in all_rows:
        if row["split"] == split:
            result.append(row)
    return result

def split_class_counts(rows):
    """
    rows: danh sách ManifestRow.
    Trả về: dict {split: {label: count}}.
    """
    counts = {}
    for split in config.SPLITS:
        counts[split] = {}
        for label in config.CLASS_NAMES:
            counts[split][label] = 0
    for row in rows:
        counts[row.split][row.label] += 1
    return counts

def duplicate_count(rows, field_name):
    """
    rows: danh sách ManifestRow.
    field_name: tên trường cần kiểm tra trùng, ví dụ "raw_sha256".
    tổng số bản ghi thuộc các nhóm trùng (count > 1).
    """
    counts = Counter()
    for row in rows:
        value = getattr(row, field_name)
        counts[value] += 1
    total_dup = 0
    for count in counts.values():
        if count > 1:
            total_dup += count
    return total_dup
def dimension_summary(rows):
    """
    rows: danh sách ManifestRow.
    dict {"WxH": count, ...} sắp xếp theo số lượng giảm dần.
    """
    counts = Counter()
    for row in rows:
        key = "{}x{}".format(row.width, row.height)
        counts[key] += 1
    most_common = counts.most_common()
    result = {}
    for key, count in most_common:
        result[key] = count
    return result

def write_report(rows):
    """
    Ghi báo cáo xử lý dữ liệu dưới dạng JSON và Markdown.
    """
    counts = split_class_counts(rows)
    notes_counter = Counter()
    for row in rows:
        notes_counter[row.note] += 1
    status_counter = Counter()
    for row in rows:
        status_counter[row.status] += 1
    exact_target_count = 0
    for row in rows:
        if (row.width, row.height) == config.IMAGE_SIZE:
            exact_target_count += 1
    summary = {
        "dataset_root": external_relative(config.DATASET_ROOT),
        "processed_data_dir": project_relative(config.PROCESSED_DATA_DIR),
        "manifest": project_relative(config.SPLIT_CSV_PATH),
        "image_size": config.IMAGE_SIZE,
        "resize_policy": config.RESIZE_POLICY,
        "total_images": len(rows),
        "split_class_counts": counts,
        "status_counts": dict(status_counter),
        "raw_exact_target_size_count": exact_target_count,
        "raw_non_target_size_count": len(rows) - exact_target_count,
        "raw_duplicate_files_by_sha256": duplicate_count(rows, "raw_sha256"),
        "processed_duplicate_files_by_sha256": duplicate_count(
            rows, "processed_sha256"
        ),
        "top_raw_dimensions": dimension_summary(rows),
        "notes": dict(notes_counter),
    }

    if not config.PROCESSING_SUMMARY_PATH.parent.exists():
        config.PROCESSING_SUMMARY_PATH.parent.mkdir(parents=True)
    with config.PROCESSING_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    lines = []
    lines.append("# Data processing report")
    lines.append("")
    lines.append(
        "Báo cáo: `python -m src.data.data_loader`."
    )
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append(
        "- Thư mục dataset thô: `{}`".format(
            external_relative(config.DATASET_ROOT)
        )
    )
    lines.append(
        "- Thư mục dataset đã xử lý: `{}`".format(
            project_relative(config.PROCESSED_DATA_DIR)
        )
    )
    lines.append(
        "- File manifest: `{}`".format(
            project_relative(config.SPLIT_CSV_PATH)
        )
    )
    lines.append(
        "- Kích thước ảnh đích: `{}x{}`".format(
            config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]
        )
    )
    lines.append("- Chính sách resize: `{}`".format(config.RESIZE_POLICY))
    lines.append("- Giữ nguyên split train/val/test ban đầu.")
    lines.append(
        "- `Forest Fire_Tester` không được dùng cho huấn luyện và đánh giá cuối."
    )
    lines.append(
        "- Không lưu augmentation lên disk. "
        "Augmentation chỉ áp dụng trong quá trình huấn luyện (train-only)."
    )
    lines.append("")
    lines.append("## Split/class counts")
    lines.append("")
    lines.append(
        "| Split | fire | nofire | smoke | smokefire | Total |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")

    for split in config.SPLITS:
        total = sum(counts[split].values())
        line = (
            "| {split} | {fire} | {nofire} | {smoke} | {smokefire} | {total} |"
        ).format(
            split=split,
            fire=counts[split]["fire"],
            nofire=counts[split]["nofire"],
            smoke=counts[split]["smoke"],
            smokefire=counts[split]["smokefire"],
            total=total,
        )
        lines.append(line)
    lines.append("")
    lines.append("## Quality checks")
    lines.append("")
    lines.append(
        "- Tổng số ảnh thô được đưa vào: {}".format(len(rows))
    )
    lines.append(
        "- Số ảnh thô đã đúng kích thước đích: {}".format(exact_target_count)
    )
    lines.append(
        "- Số ảnh thô phải resize về kích thước đích: {}".format(
            len(rows) - exact_target_count
        )
    )
    lines.append(
        "- Số file thô trùng chính xác theo SHA256: {}".format(
            summary["raw_duplicate_files_by_sha256"]
        )
    )
    lines.append(
        "- Số file đã xử lý trùng chính xác theo SHA256: {}".format(
            summary["processed_duplicate_files_by_sha256"]
        )
    )
    lines.append(
        "- Phân bố trạng thái: `{}`".format(dict(status_counter))
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Dataset thô có kích thước ảnh tương quan với class."
    )
    lines.append(
        "- Dùng resize trực tiếp để tránh thêm padding (có thể trở thành shortcut feature)."
    )
    lines.append(
        "- Ảnh validation và test được xử lý độc lập, không áp dụng augmentation."
    )
    lines.append(
        "- Cây thư mục processed giữ nguyên cấu trúc split/class của bản gốc."
    )
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    lines.append("- `data/split.csv`")
    lines.append("- `data/processed/processing_summary.json`")
    lines.append("- `data/processed/processing_report.md`")
    lines.append("- `data/processed/samples/train_grid.jpg`")
    lines.append("- `data/processed/samples/val_grid.jpg`")
    lines.append("- `data/processed/samples/test_grid.jpg`")
    if not config.PROCESSING_REPORT_PATH.parent.exists():
        config.PROCESSING_REPORT_PATH.parent.mkdir(parents=True)

    config.PROCESSING_REPORT_PATH.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

def write_sample_grids(rows, samples_per_class=4):
    """
    rows: danh sách ManifestRow.
    samples_per_class: số ảnh mẫu lấy tối đa cho mỗi class trong mỗi split.
    """
    sample_dir = config.PROCESSED_DATA_DIR / "samples"
    if not sample_dir.exists():
        sample_dir.mkdir(parents=True)
    by_split_class = defaultdict(list)
    for row in rows:
        key = (row.split, row.label)
        by_split_class[key].append(row)

    tile_width, tile_height = config.IMAGE_SIZE
    label_height = 26
    padding = 8

    for split in config.SPLITS:
        grid_width = (
            samples_per_class * tile_width
            + (samples_per_class + 1) * padding
        )
        grid_height = (
            len(config.CLASS_NAMES) * (tile_height + label_height)
            + (len(config.CLASS_NAMES) + 1) * padding
        )

        grid = Image.new("RGB", (grid_width, grid_height), "white")
        draw = ImageDraw.Draw(grid)

        for class_index, label in enumerate(config.CLASS_NAMES):
            y = padding + class_index * (
                tile_height + label_height + padding
            )
            draw.text(
                (padding, y + 4),
                "{} / {}".format(split, label),
                fill=(20, 20, 20),
            )
            sample_list = by_split_class[(split, label)]
            for sample_index, row in enumerate(
                sample_list[:samples_per_class]
            ):
                x = padding + sample_index * (tile_width + padding)
                img_path = config.PROJECT_ROOT / row.processed_filepath
                with Image.open(img_path) as image:
                    grid.paste(image.convert("RGB"), (x, y + label_height))

        grid_path = sample_dir / "{}_grid.jpg".format(split)
        grid.save(grid_path, format="JPEG", quality=92)

def console_safe(value):
    s = str(value)
    return s.encode("ascii", "backslashreplace").decode("ascii")

def main():
    rows = prepare_dataset()
    print("Đã chuẩn bị {} ảnh.".format(len(rows)))
    print(
        "File manifest: {}".format(console_safe(config.SPLIT_CSV_PATH))
    )
    print(
        "File báo cáo: {}".format(
            console_safe(config.PROCESSING_REPORT_PATH)
        )
    )

if __name__ == "__main__":
    main()