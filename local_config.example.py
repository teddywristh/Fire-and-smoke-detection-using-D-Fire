"""Example local path configuration.

Copy this file to `local_config.py` and edit the paths for your own machine.
`local_config.py` is ignored by Git, so each team member can keep a different
dataset location without changing tracked project files.
"""

from pathlib import Path


# Point this to the folder that directly contains train/val/test.
DATASET_ROOT = Path(r"../dataset/Forect Fire/Forest Fire_Dataset")

# Optional: only needed for manual demo images.
TESTER_ROOT = Path(r"../dataset/Forect Fire/Forest Fire_Tester")

# Optional: keep processed outputs inside the project by default.
# DATA_DIR = Path("data")
