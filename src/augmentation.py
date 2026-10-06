"""Training augmentation utilities.

Augmentation must be applied only to the training split and should not be saved
over validation or test data. Keep validation/test deterministic.

`src.datasets.make_dataset(..., training=True)` applies `build_augmentation()`
to every training batch, so all models share one policy.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

import config


def describe_recommended_augmentation() -> dict[str, object]:
    """Return the conservative augmentation policy planned for training."""
    return {
        "horizontal_flip": True,
        "rotation_range_degrees": 10,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "zoom_range": 0.1,
        "brightness_range": [0.8, 1.2],
        "apply_to": "train_only",
    }


def build_augmentation(seed: int = config.RANDOM_SEED) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return a function that augments a batch of [0, 255] float images."""
    policy = describe_recommended_augmentation()
    geometric = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(
                policy["rotation_range_degrees"] / 360, fill_mode="reflect", seed=seed
            ),
            tf.keras.layers.RandomTranslation(
                policy["height_shift_range"],
                policy["width_shift_range"],
                fill_mode="reflect",
                seed=seed,
            ),
            tf.keras.layers.RandomZoom(policy["zoom_range"], fill_mode="reflect", seed=seed),
        ],
        name="train_augmentation",
    )
    low, high = policy["brightness_range"]

    def augment(images: tf.Tensor) -> tf.Tensor:
        images = geometric(images, training=True)
        # Multiplicative brightness, as in Keras ImageDataGenerator(brightness_range=...).
        factors = tf.random.uniform([tf.shape(images)[0], 1, 1, 1], low, high)
        return tf.clip_by_value(images * factors, 0.0, 255.0)

    return augment
