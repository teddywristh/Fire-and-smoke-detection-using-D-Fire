"""Project-wide settings.

The dataset location differs per machine. Set it without editing this file:
copy `local_config.example.py` to `local_config.py` (ignored by Git), or set
the `FOREST_FIRE_DATASET_ROOT` environment variable.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    import local_config as _local
except ModuleNotFoundError:
    _local = None

_default_root = PROJECT_ROOT.parent / "dataset" / "Forect Fire" / "Forest Fire_Dataset"
DATASET_ROOT = Path(
    os.environ.get("FOREST_FIRE_DATASET_ROOT") or getattr(_local, "DATASET_ROOT", _default_root)
).expanduser()
if not DATASET_ROOT.is_absolute():
    DATASET_ROOT = (PROJECT_ROOT / DATASET_ROOT).resolve()

SPLIT_CSV = PROJECT_ROOT / "data" / "split_aspect.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["fire", "nofire", "smoke", "smokefire"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42
