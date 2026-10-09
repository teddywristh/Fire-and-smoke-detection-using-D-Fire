"""Shared data pipeline: data/split.csv -> tf.data for every model.

Every model receives the same tensors:

- images: float32, shape (224, 224, 3), pixel values in [0, 255]
- labels: one-hot float32, shape (4,), in the order of `config.CLASS_NAMES`

Models do their own resizing and scaling as their first layers, so a saved
checkpoint works directly on processed images. Augmentation is applied to the
training split only (see `src/augmentation.py`). Validation and test keep the
manifest order, so predictions line up with the rows of their frame.

Typical use in a model file:

    frames, datasets = load_datasets()
    model.fit(datasets["train"], validation_data=datasets["val"], ...)
    probabilities = model.predict(datasets["test"])
    y_true = frames["test"]["class_index"].to_numpy()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf

import config
from src.augmentation import build_augmentation


NUM_CLASSES = len(config.CLASS_NAMES)
# config.IMAGE_SIZE is (width, height); TensorFlow shapes are (height, width, channels).
INPUT_SHAPE = (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3)
BATCH_SIZE = 32

_REQUIRED_COLUMNS = {"processed_filepath", "split", "label", "class_index", "status"}


def load_frames(manifest_path=config.SPLIT_CSV_PATH) -> dict[str, pd.DataFrame]:
    """Read the manifest and return one frame per split, rows with status=ok only."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run: python -m src.data_loader")
    frame = pd.read_csv(manifest_path)
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"split.csv is missing columns: {sorted(missing)}")

    frame = frame[frame["status"] == "ok"].copy()
    frame["class_index"] = frame["class_index"].astype(int)
    if not frame["label"].map(config.CLASS_TO_INDEX).equals(frame["class_index"]):
        raise ValueError("split.csv labels do not match config.CLASS_TO_INDEX")
    frame["path"] = [str(config.PROJECT_ROOT / path) for path in frame["processed_filepath"]]

    frames = {split: frame[frame["split"] == split].reset_index(drop=True) for split in config.SPLITS}
    for split, part in frames.items():
        if part.empty:
            raise ValueError(f"Split '{split}' has no images")
        first_image = Path(part["path"].iloc[0])
        if not first_image.exists():
            raise FileNotFoundError(f"{first_image} not found. Run: python -m src.data_loader")
    return frames


def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.ensure_shape(tf.cast(image, tf.float32), INPUT_SHAPE)
    return image, tf.one_hot(label, NUM_CLASSES)


def make_dataset(
    frame: pd.DataFrame,
    training: bool = False,
    batch_size: int = BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tf.data.Dataset:
    """Build a batched dataset; shuffle and augment only when training=True."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (frame["path"].to_numpy(), frame["class_index"].to_numpy())
    )
    if training:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(_decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        augment = build_augmentation(seed)
        dataset = dataset.map(
            lambda images, labels: (augment(images), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_datasets(
    batch_size: int = BATCH_SIZE,
) -> tuple[dict[str, pd.DataFrame], dict[str, tf.data.Dataset]]:
    """Return manifest frames and datasets for train (augmented), val and test."""
    frames = load_frames()
    datasets = {
        split: make_dataset(frames[split], training=(split == "train"), batch_size=batch_size)
        for split in config.SPLITS
    }
    return frames, datasets
