"""Compatibility shim: the shared implementation lives in `src/preprocessing.py`."""

from src.preprocessing import *  # noqa: F401,F403
from src.preprocessing import ImageMetadata, read_image_metadata, save_processed_image  # noqa: F401
