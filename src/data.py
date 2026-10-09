"""The one data pipeline used by every task (the transfer-learning branch protocol).

1. Split, `data/split_aspect.csv`: the Kaggle test split is kept untouched. The
   Kaggle train+val pool is re-split 800/200 per class, stratified by aspect
   group, so validation also holds wide `smokefire` images. Rebuild it from the
   raw images with `python -m src.data`.
2. Geometry: raw JPEG -> square crop -> bilinear resize to 224x224. Train takes a
   random crop (side 72-100% of the short side, random position); val/test take
   the center crop. No stretching and no padding, so the aspect ratio, which is
   strongly tied to the class in this dataset, does not leak into the pixels.
3. Train-only augmentation: small affine (rotate +-5 deg, translate +-4%, scale
   0.94-1.06, shear +-2 deg), horizontal flip, brightness and contrast x0.88-1.12.
4. Train sampling: each epoch draws len(train) images with replacement, with
   probability proportional to 1/sqrt(size of the image's (class, aspect group)).

Batches hold float32 pixels in [0, 255] and one-hot labels. Every model rescales
the pixels in its own first layers.
"""

import hashlib
import math
import os
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import config

NUM_CLASSES = len(config.CLASS_NAMES)
FILENAME = re.compile(r"^(fire|nofire|smoke|smokefire)_(train|val|test)_\d+\.jpg$")


def aspect_group(width, height):
    ratio = width / height
    if 0.9 <= ratio <= 1.1:
        return "square"
    if 1.6 <= ratio <= 1.95:
        return "wide"
    return "other"


# ---------------------------------------------------------------- manifest

def scan_raw_images():
    """One row per raw image, in Kaggle split/class/file-name order."""
    rows = []
    for split in config.SPLITS:
        for label in config.CLASS_NAMES:
            for path in sorted((config.DATASET_ROOT / split / label).glob("*.jpg")):
                match = FILENAME.match(path.name)
                if not match or match.groups() != (label, split):
                    raise ValueError(f"Unexpected file name: {path}")
                with Image.open(path) as image:
                    width, height, image_format = image.width, image.height, image.format
                    image.verify()
                if image_format != "JPEG":
                    raise ValueError(f"Not a JPEG file: {path}")
                rows.append({
                    "filepath": path.relative_to(config.DATASET_ROOT).as_posix(),
                    "source_split": split,
                    "label": label,
                    "class_index": config.CLASS_TO_INDEX[label],
                    "width": width,
                    "height": height,
                    "aspect_group": aspect_group(width, height),
                    "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
    frame = pd.DataFrame(rows)
    if frame.raw_sha256.duplicated().any():
        raise ValueError("The raw dataset contains byte-identical duplicates")
    return frame


def _validation_quotas(class_frame, fraction, min_wide):
    """Validation images per aspect group for one class (kept from create_aspect_split)."""
    groups = class_frame.aspect_group
    counts = {group: int((groups == group).sum()) for group in groups.unique()}
    target = round(len(class_frame) * fraction)
    exact = {group: count * fraction for group, count in counts.items()}
    quotas = {group: int(value) for group, value in exact.items()}
    order = sorted(counts, key=lambda g: (exact[g] - quotas[g], counts[g]), reverse=True)
    while sum(quotas.values()) < target:
        quotas[next(g for g in order if quotas[g] < counts[g])] += 1
    while sum(quotas.values()) > target:
        quotas[next(g for g in reversed(order) if quotas[g] > 0)] -= 1
    wanted = min(min_wide, counts.get("wide", 0))
    if quotas.get("wide", 0) < wanted:
        missing = wanted - quotas["wide"]
        quotas["wide"] = wanted
        for group in sorted((g for g in counts if g != "wide"), key=counts.get, reverse=True):
            take = min(missing, quotas[group])
            quotas[group] -= take
            missing -= take
    return quotas


def build_manifest(fraction=0.2, min_wide_smokefire_val=4, seed=config.RANDOM_SEED):
    """Re-split the Kaggle train+val pool; the Kaggle test split stays the test set."""
    frame = scan_raw_images()
    pool = frame[frame.source_split != "test"]
    rng = random.Random(seed)
    val_index, group_number = [], 0
    for label in sorted(pool.label.unique()):
        class_frame = pool[pool.label == label]
        min_wide = min_wide_smokefire_val if label == "smokefire" else 0
        quotas = _validation_quotas(class_frame, fraction, min_wide)
        for group in sorted(quotas):
            index = class_frame.index[class_frame.aspect_group == group].tolist()
            rng.seed(seed + group_number)
            rng.shuffle(index)
            val_index += index[:quotas[group]]
            group_number += 1
    frame.insert(1, "split", np.where(frame.source_split == "test", "test", "train"))
    frame.loc[val_index, "split"] = "val"
    order = frame.split.map({split: i for i, split in enumerate(config.SPLITS)})
    return frame.iloc[np.argsort(order.to_numpy(), kind="stable")].reset_index(drop=True)


# ---------------------------------------------------------------- tf.data

def load_frames(path=config.SPLIT_CSV):
    """Manifest rows per split, with absolute image paths."""
    frame = pd.read_csv(path)
    frame["path"] = [str(config.DATASET_ROOT / p) for p in frame.filepath]
    missing = [p for p in frame.path if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[0]}; check config.DATASET_ROOT")
    return {split: frame[frame.split == split].reset_index(drop=True) for split in config.SPLITS}


def sampling_probabilities(frame):
    counts = frame.groupby(["label", "aspect_group"]).label.transform("size").to_numpy()
    weights = 1.0 / np.sqrt(counts)
    return weights / weights.sum()


def _random_affine(image):
    size = config.IMAGE_SIZE
    uniform = lambda low, high: tf.random.uniform([], low, high)
    angle = uniform(-5.0, 5.0) * math.pi / 180
    shear_x, shear_y = uniform(-2.0, 2.0) * math.pi / 180, uniform(-2.0, 2.0) * math.pi / 180
    scale = uniform(0.94, 1.06)
    tx, ty = tf.round(size * uniform(-0.04, 0.04)), tf.round(size * uniform(-0.04, 0.04))
    rotation = tf.reshape(tf.stack([tf.cos(angle), -tf.sin(angle), tf.sin(angle), tf.cos(angle)]), [2, 2])
    shear = tf.reshape(tf.stack([1.0, tf.tan(shear_x), tf.tan(shear_y), 1.0]), [2, 2])
    inverse = tf.linalg.inv(scale * rotation @ shear)
    # ImageProjectiveTransform maps each output pixel back to its input pixel.
    center = (size - 1) / 2
    offset = tf.stack([center + tx, center + ty])
    shift = tf.stack([center, center]) - tf.linalg.matvec(inverse, offset)
    transform = tf.stack([inverse[0, 0], inverse[0, 1], shift[0],
                          inverse[1, 0], inverse[1, 1], shift[1], 0.0, 0.0])
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None], transforms=transform[None], output_shape=[size, size],
        fill_value=0.0, interpolation="BILINEAR", fill_mode="CONSTANT")[0]


def _augment(image):
    image = _random_affine(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.clip_by_value(image * tf.random.uniform([], 0.88, 1.12), 0.0, 255.0)
    # PIL-style contrast: blend with the mean grey level of the image.
    mean = tf.reduce_mean(tf.image.rgb_to_grayscale(image))
    return tf.clip_by_value(mean + tf.random.uniform([], 0.88, 1.12) * (image - mean), 0.0, 255.0)


def _load(path, height, width, label, training):
    if training:
        short = tf.cast(tf.minimum(height, width), tf.float32)
        side = tf.maximum(1, tf.cast(tf.round(short * tf.random.uniform([], 0.72, 1.0)), tf.int32))
        top = tf.random.uniform([], 0, height - side + 1, dtype=tf.int32)
        left = tf.random.uniform([], 0, width - side + 1, dtype=tf.int32)
    else:
        side = tf.minimum(height, width)
        top, left = (height - side) // 2, (width - side) // 2
    image = tf.io.decode_and_crop_jpeg(tf.io.read_file(path), tf.stack([top, left, side, side]), channels=3)
    image = tf.image.resize(image, [config.IMAGE_SIZE, config.IMAGE_SIZE], antialias=True)
    if training:
        image = _augment(image)
    return image, tf.one_hot(label, NUM_CLASSES)


def make_dataset(frame, training=False, batch_size=config.BATCH_SIZE, seed=config.RANDOM_SEED):
    """Batched dataset; val/test keep the manifest order so predictions line up with `frame`."""
    columns = (frame.path.to_numpy(), frame.height.to_numpy("int32"),
               frame.width.to_numpy("int32"), frame.class_index.to_numpy("int32"))
    if training:
        rng = np.random.default_rng(seed)
        probabilities = sampling_probabilities(frame)
        tensors = [tf.constant(column) for column in columns]
        draws = tf.data.Dataset.from_generator(
            lambda: rng.choice(len(frame), size=len(frame), p=probabilities),
            output_signature=tf.TensorSpec([], tf.int64))
        dataset = draws.map(lambda i: tuple(tf.gather(t, i) for t in tensors))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(columns)
    dataset = dataset.map(lambda *row: _load(*row, training=training), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_datasets(batch_size=config.BATCH_SIZE):
    """Frames and datasets for train (sampled, augmented), val and test."""
    frames = load_frames()
    datasets = {split: make_dataset(frames[split], training=(split == "train"), batch_size=batch_size)
                for split in config.SPLITS}
    return frames, datasets


if __name__ == "__main__":
    manifest = build_manifest()
    config.SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.SPLIT_CSV, index=False, lineterminator="\n")
    print(f"Wrote {config.SPLIT_CSV} ({len(manifest)} images)")
    print(pd.crosstab([manifest.split, manifest.label], manifest.aspect_group))
