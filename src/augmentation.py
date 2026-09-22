"""Training augmentation utilities.

Augmentation must be applied only to the training split and should not be saved
over validation or test data. Keep validation/test deterministic.
"""


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
