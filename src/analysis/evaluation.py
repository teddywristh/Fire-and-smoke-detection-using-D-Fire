"""Reusable evaluation helpers for the forest-fire classifiers.

The functions in this module operate on true and predicted class indexes.
They are intentionally independent of a specific model or DataLoader so all
training pipelines can produce the same evaluation artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import config


DEFAULT_CLASS_NAMES = tuple(config.CLASS_NAMES)


def _as_class_array(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    """Convert class indexes to a one-dimensional integer NumPy array."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer class indexes")
    return array.astype(int, copy=False)


def _validate_inputs(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    true_array = _as_class_array(y_true, "y_true")
    predicted_array = _as_class_array(y_pred, "y_pred")
    names = tuple(class_names)
    if len(true_array) != len(predicted_array):
        raise ValueError("y_true and y_pred must have the same length")
    if not names:
        raise ValueError("class_names must contain at least one class")
    if np.any(true_array < 0) or np.any(true_array >= len(names)):
        raise ValueError("y_true contains a class index outside class_names")
    if np.any(predicted_array < 0) or np.any(predicted_array >= len(names)):
        raise ValueError("y_pred contains a class index outside class_names")
    return true_array, predicted_array, names


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> dict[str, object]:
    """Return a JSON-serializable classification report."""
    true_array, predicted_array, names = _validate_inputs(y_true, y_pred, class_names)
    return classification_report(
        true_array,
        predicted_array,
        labels=list(range(len(names))),
        target_names=list(names),
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_frame(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> pd.DataFrame:
    """Return a labeled confusion matrix with true classes as rows."""
    true_array, predicted_array, names = _validate_inputs(y_true, y_pred, class_names)
    matrix = confusion_matrix(
        true_array,
        predicted_array,
        labels=list(range(len(names))),
    )
    return pd.DataFrame(matrix, index=names, columns=names)


def misclassification_frame(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    sample_ids: Sequence[str] | None = None,
    probabilities: np.ndarray | None = None,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> pd.DataFrame:
    """Return one row for every incorrect prediction.

    ``probabilities`` may contain one score per class and sample. When
    provided, the predicted-class confidence is included in the result.
    """
    true_array, predicted_array, names = _validate_inputs(y_true, y_pred, class_names)
    if sample_ids is None:
        ids = np.arange(len(true_array)).astype(str)
    else:
        ids = np.asarray(sample_ids)
        if ids.ndim != 1 or len(ids) != len(true_array):
            raise ValueError("sample_ids must have one value per prediction")

    frame = pd.DataFrame(
        {
            "sample_id": ids,
            "true_index": true_array,
            "true_label": [names[index] for index in true_array],
            "predicted_index": predicted_array,
            "predicted_label": [names[index] for index in predicted_array],
        }
    )
    if probabilities is not None:
        scores = np.asarray(probabilities)
        if scores.shape != (len(true_array), len(names)):
            raise ValueError(
                "probabilities must have shape "
                f"({len(true_array)}, {len(names)})"
            )
        frame["predicted_confidence"] = scores[
            np.arange(len(predicted_array)), predicted_array
        ]
    return frame.loc[frame["true_index"] != frame["predicted_index"]].reset_index(
        drop=True
    )


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save an annotated confusion-matrix image."""
    if matrix.index.tolist() != matrix.columns.tolist():
        raise ValueError("matrix index and columns must contain the same classes")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(matrix.columns)),
        yticks=range(len(matrix.index)),
        xticklabels=matrix.columns,
        yticklabels=matrix.index,
    )
    threshold = matrix.to_numpy().max() / 2 if matrix.size else 0
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iat[row, column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    output_dir: str | Path,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    sample_ids: Sequence[str] | None = None,
    probabilities: np.ndarray | None = None,
    title: str = "Confusion Matrix",
) -> dict[str, object]:
    """Create the standard report, matrix, plot, and error-analysis CSV."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = classification_metrics(y_true, y_pred, class_names)
    matrix = confusion_matrix_frame(y_true, y_pred, class_names)
    errors = misclassification_frame(
        y_true,
        y_pred,
        sample_ids=sample_ids,
        probabilities=probabilities,
        class_names=class_names,
    )

    with (output / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    matrix.to_csv(output / "confusion_matrix.csv", index_label="true/predicted")
    errors.to_csv(output / "misclassifications.csv", index=False)
    plot_confusion_matrix(matrix, output / "confusion_matrix.png", title)
    return {"classification_report": report, "confusion_matrix": matrix, "errors": errors}
