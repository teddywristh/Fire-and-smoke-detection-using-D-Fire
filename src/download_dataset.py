"""Download Forest Fire C4 from Kaggle and prepare the shared dataset manifest.

Run with ``python -m src.download_dataset``. KaggleHub stores the downloaded
archive in ``data/raw/kagglehub`` so the project's data stays in one place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Support both `python -m src.download_dataset` and direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


DATASET_HANDLE = "obulisainaren/forest-fire-c4"


def find_dataset_root(download_path: Path) -> Path:
    """Find the directory with the expected split/class folder structure."""
    expected = [
        download_path / split / class_name
        for split in config.SPLITS
        for class_name in config.CLASS_NAMES
    ]
    if all(path.is_dir() for path in expected):
        return download_path

    for candidate in download_path.rglob("train"):
        if not candidate.is_dir():
            continue
        root = candidate.parent
        if all((root / split / class_name).is_dir()
               for split in config.SPLITS for class_name in config.CLASS_NAMES):
            return root

    raise FileNotFoundError(
        f"Downloaded files at {download_path} do not contain the expected "
        f"train/val/test folders with classes: {', '.join(config.CLASS_NAMES)}."
    )


def main() -> None:
    cache = config.DATA_DIR / "raw" / "kagglehub"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache.resolve())

    try:
        import kagglehub
    except ImportError as error:
        raise SystemExit("Install KaggleHub first: `pip install kagglehub`") from error

    downloaded = Path(kagglehub.dataset_download(DATASET_HANDLE)).resolve()
    dataset_root = find_dataset_root(downloaded)
    print(f"Dataset downloaded to: {downloaded}")
    print(f"Dataset root: {dataset_root}")

    # data_loader uses config.DATASET_ROOT as the source and writes processed
    # images plus split.csv using the project's established pipeline.
    config.DATASET_ROOT = dataset_root
    from src.data_loader import prepare_dataset

    rows = prepare_dataset()
    print(f"Prepared {len(rows)} image records.")
    print(f"Manifest: {config.SPLIT_CSV_PATH}")


if __name__ == "__main__":
    main()
