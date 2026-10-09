"""Compatibility shim: the shared implementation lives in `src/data_loader.py`.

`python -m src.data.data_loader` and `python -m src.data_loader` are the same
command; both write `data/split.csv` with `filepath` relative to `DATASET_ROOT`.
"""

from src.data_loader import *  # noqa: F401,F403
from src.data_loader import main, prepare_dataset  # noqa: F401

if __name__ == "__main__":
    main()
