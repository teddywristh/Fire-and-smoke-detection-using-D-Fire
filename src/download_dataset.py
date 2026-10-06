# -*- coding: utf-8 -*-
"""
Tải bộ dữ liệu Forest Fire C4 từ Kaggle và tạo file manifest chung cho dự án.
Chạy bằng lệnh: python -m src.download_dataset
"""

from __future__ import annotation
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config
DATASET_HANDLE = "obulisainaren/forest-fire-c4"

def find_dataset_root(download_path):
    """
    Tìm thư mục gốc của dataset sau khi tải về, dựa trên cấu trúc
    split/class mà config yêu cầu.
    download_path: đường dẫn tới nơi KaggleHub giải nén dataset.
    Trả về: đường dẫn tới thư mục gốc có cấu trúc train/val/test/classes.
    """
    expected_dirs = []
    for split in config.SPLITS:
        for class_name in config.CLASS_NAMES:
            expected_dirs.append(download_path / split / class_name)
    all_exist = True
    for p in expected_dirs:
        if not p.is_dir():
            all_exist = False
            break
    if all_exist:
        return download_path
    for candidate in download_path.rglob("train"):
        if not candidate.is_dir():
            continue
        root = candidate.parent
        ok = True
        for split in config.SPLITS:
            for class_name in config.CLASS_NAMES:
                if not (root / split / class_name).is_dir():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return root
    msg = (
        "Các file tải về tại {} không chứa thư mục "
        "train/val/test với các lớp: {}."
    ).format(str(download_path), ", ".join(config.CLASS_NAMES))
    raise FileNotFoundError(msg)


def main():
    cache = config.DATA_DIR / "raw" / "kagglehub"
    if not cache.exists():
        cache.mkdir(parents=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache.resolve())
    try:
        import kagglehub
    except ImportError as error:
        msg = "Chưa cài đặt KaggleHub. Cài đặt bằng lệnh: pip install kagglehub"
        raise SystemExit(msg) from error

    downloaded_str = kagglehub.dataset_download(DATASET_HANDLE)
    downloaded = Path(downloaded_str).resolve()
    dataset_root = find_dataset_root(downloaded)

    print("Dataset: {}".format(str(downloaded)))
    print("Root folder của dataset: {}".format(str(dataset_root)))

    config.DATASET_ROOT = dataset_root

    from src.data_loader import prepare_dataset
    rows = prepare_dataset()

    print("Đã chuẩn bị {} ảnh.".format(len(rows)))
    print("File manifest: {}".format(str(config.SPLIT_CSV_PATH)))


if __name__ == "__main__":
    main()