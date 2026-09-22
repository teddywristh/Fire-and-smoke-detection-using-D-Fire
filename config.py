"""Central configuration for the forest fire classification project.

Team members can keep their own dataset path without editing tracked files:

1. Copy `local_config.example.py` to `local_config.py`.
2. Change `DATASET_ROOT` in `local_config.py`.

Environment variables override `local_config.py` when present:

- `FOREST_FIRE_DATASET_ROOT`
- `FOREST_FIRE_TESTER_ROOT`
- `FOREST_FIRE_DATA_DIR`
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

try:
    import local_config as _local_config
except ModuleNotFoundError:
    _local_config = None


def _local_value(name: str, default: object) -> object:
    if _local_config is None:
        return default
    return getattr(_local_config, name, default)


def _resolve_path(value: object) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _path_setting(env_name: str, local_name: str, default: object) -> Path:
    value = os.environ.get(env_name)
    if value is None:
        value = _local_value(local_name, default)
    return _resolve_path(value)


DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "dataset" / "Forect Fire" / "Forest Fire_Dataset"
DEFAULT_TESTER_ROOT = PROJECT_ROOT.parent / "dataset" / "Forect Fire" / "Forest Fire_Tester"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

DATASET_ROOT = _path_setting("FOREST_FIRE_DATASET_ROOT", "DATASET_ROOT", DEFAULT_DATASET_ROOT)
TESTER_ROOT = _path_setting("FOREST_FIRE_TESTER_ROOT", "TESTER_ROOT", DEFAULT_TESTER_ROOT)

DATA_DIR = _path_setting("FOREST_FIRE_DATA_DIR", "DATA_DIR", DEFAULT_DATA_DIR)
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_CSV_PATH = DATA_DIR / "split.csv"
PROCESSING_REPORT_PATH = PROCESSED_DATA_DIR / "processing_report.md"
PROCESSING_SUMMARY_PATH = PROCESSED_DATA_DIR / "processing_summary.json"

SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["fire", "nofire", "smoke", "smokefire"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = (224, 224)
RESIZE_POLICY = "direct_resize"
JPEG_QUALITY = 95
RANDOM_SEED = 42
