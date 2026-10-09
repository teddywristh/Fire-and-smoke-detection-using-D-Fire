"""One file per model; each owner keeps their architecture and training here.

Contract with the shared data pipeline (`src/datasets.py`):

- Input: (224, 224, 3) float32 images with pixel values in [0, 255].
- The model does its own resizing and scaling as its first layers
  (for example `Resizing` and `Rescaling(1 / 255)`, or a backbone's own
  preprocessing), so the saved checkpoint works on processed images directly.
- Output: softmax probabilities over `config.CLASS_NAMES`.
- Labels are one-hot, so compile with categorical cross-entropy.
- Augmentation comes from the pipeline (train only); do not add more inside
  the model without telling the team, or the comparison is no longer fair.

Each file saves its checkpoint to `models/<name>_best.keras` and its outputs
to `results/<name>/`, and runs with `python -m src.models.<name>`.
"""
