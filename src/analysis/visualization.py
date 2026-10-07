"""Visual reports for the current frozen Xception 224x224 experiment.

The default run directory is
``results/transfer_frozen/xception_square_aspect_balanced_224``.
It reads the existing CSV/JSON artifacts and never loads the model or test
images. Run from the project root:

    python -m src.analysis.visualization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


DEFAULT_RUN_DIR = (
    config.PROJECT_ROOT
    / "results"
    / "transfer_frozen"
    / "xception_square_aspect_balanced_224"
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "training_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing training history: {path}")
    frame = pd.read_csv(path)
    required = {
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "val_macro_f1",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"training_history.csv is missing columns: {sorted(missing)}")
    return frame


def plot_training_history(history: pd.DataFrame, output_path: Path, best_epoch: int) -> None:
    """Plot loss, accuracy, macro F1, and learning rate over epochs."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    epochs = history["epoch"]

    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Validation")
    axes[0, 0].set(title="Loss", xlabel="Epoch", ylabel="Loss")

    axes[0, 1].plot(epochs, history["train_accuracy"], label="Train")
    axes[0, 1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0, 1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")

    axes[1, 0].plot(epochs, history["val_macro_f1"], color="darkorange")
    best = history.loc[history["epoch"] == best_epoch]
    if not best.empty:
        axes[1, 0].scatter(
            best["epoch"], best["val_macro_f1"], color="crimson", zorder=3,
            label=f"Best epoch {best_epoch}",
        )
    axes[1, 0].set(title="Validation macro F1", xlabel="Epoch", ylabel="F1")

    if "learning_rate" in history:
        axes[1, 1].plot(epochs, history["learning_rate"], color="seagreen")
        axes[1, 1].set_yscale("log")
    axes[1, 1].set(title="Learning rate", xlabel="Epoch", ylabel="Learning rate")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="best")
    figure.suptitle("Frozen Xception 224x224 training history", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_class_metrics(report: dict[str, Any], output_path: Path) -> None:
    """Plot precision, recall, and F1 for each validation class."""
    classes = [
        name for name, values in report.items()
        if isinstance(values, dict) and "f1-score" in values
    ]
    metrics = ["precision", "recall", "f1-score"]
    values = np.array([[report[name][metric] for metric in metrics] for name in classes])
    positions = np.arange(len(classes))
    width = 0.24

    figure, axis = plt.subplots(figsize=(10, 6))
    for index, metric in enumerate(metrics):
        axis.bar(
            positions + (index - 1) * width,
            values[:, index],
            width,
            label=metric.replace("-", " ").title(),
        )
    axis.set(
        title="Validation metrics by class",
        ylabel="Score",
        xticks=positions,
        xticklabels=classes,
        ylim=(0, 1.05),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(csv_path: Path, output_path: Path) -> None:
    """Plot the validation confusion matrix from the existing CSV artifact."""
    matrix = pd.read_csv(csv_path, index_col=0)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, label="Images")
    axis.set(
        title="Validation confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(matrix.columns)),
        yticks=range(len(matrix.index)),
        xticklabels=matrix.columns,
        yticklabels=matrix.index,
    )
    threshold = matrix.to_numpy().max() / 2
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iat[row, column]
            axis.text(
                column, row, str(value), ha="center", va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_aspect_analysis(aspect_metrics: dict[str, Any], output_path: Path) -> None:
    """Plot accuracy, supported-class F1, and support by aspect group."""
    groups = list(aspect_metrics)
    accuracy = [aspect_metrics[group]["accuracy"] for group in groups]
    macro_f1 = [
        aspect_metrics[group]["macro_f1_supported_classes"] for group in groups
    ]
    support = [aspect_metrics[group]["support"] for group in groups]
    positions = np.arange(len(groups))

    figure, (score_axis, support_axis) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]}
    )
    width = 0.34
    score_axis.bar(positions - width / 2, accuracy, width, label="Accuracy")
    score_axis.bar(positions + width / 2, macro_f1, width, label="Supported-class macro F1")
    score_axis.set(
        title="Validation performance by aspect group",
        xticks=positions,
        xticklabels=groups,
        ylim=(0, 1.05),
        ylabel="Score",
    )
    score_axis.grid(axis="y", alpha=0.25)
    score_axis.legend()

    bars = support_axis.bar(groups, support, color=["steelblue", "darkorange", "slategray"])
    support_axis.set(title="Validation support", ylabel="Images")
    support_axis.grid(axis="y", alpha=0.25)
    support_axis.bar_label(bars)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_validation_analysis(
    summary: dict[str, Any],
    report: dict[str, Any],
    aspect_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a factual Markdown interpretation of validation artifacts."""
    class_rows = []
    for name, values in report.items():
        if isinstance(values, dict) and "f1-score" in values:
            class_rows.append(
                "| {} | {:.2%} | {:.2%} | {:.2%} | {} |".format(
                    name,
                    values["precision"],
                    values["recall"],
                    values["f1-score"],
                    int(values["support"]),
                )
            )
    aspect_rows = []
    for group, values in aspect_metrics.items():
        aspect_rows.append(
            "| {} | {} | {:.2%} | {:.2%} |".format(
                group,
                values["support"],
                values["accuracy"],
                values["macro_f1_supported_classes"],
            )
        )
    output_path.write_text(
        "\n".join(
            [
                "# Validation analysis: frozen Xception 224x224",
                "",
                "This report is generated from validation artifacts only. "
                "It does not load the checkpoint or evaluate test images.",
                "",
                "## Protocol",
                "",
                f"- Manifest: `{summary['split_manifest']}`",
                f"- Crop mode: `{summary['crop_mode']}`",
                f"- Aspect-balanced sampling: `{summary['aspect_balanced_sampling']}`",
                f"- Best epoch: `{summary['best_epoch']}`",
                "- Selection metric: validation macro F1",
                "- Test evaluation: disabled",
                "",
                "## Overall validation result",
                "",
                f"- Accuracy: **{summary['validation_accuracy']:.2%}**",
                f"- Macro F1: **{summary['validation_macro_f1']:.2%}**",
                "",
                "## Class-level metrics",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
                *class_rows,
                "",
                "## Aspect-group analysis",
                "",
                "| Group | Support | Accuracy | Supported-class macro F1 |",
                "|---|---:|---:|---:|",
                *aspect_rows,
                "",
                "The wide and other groups contain fewer samples and a different "
                "class composition than the square group. Their scores should be "
                "used for diagnosis, not as standalone claims of camera-domain "
                "robustness. In particular, wide `smokefire` support is too small "
                "for a reliable estimate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def create_visual_report(run_dir: str | Path = DEFAULT_RUN_DIR) -> list[Path]:
    """Create all visual artifacts from one frozen-Xception run directory."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    summary = _load_json(run_path / "summary.json")
    report = _load_json(run_path / "validation_classification_report.json")
    aspect_metrics = _load_json(run_path / "validation_aspect_group_metrics.json")
    history = _load_history(run_path)
    experiment_config = run_path / "experiment_config.json"
    if not experiment_config.exists():
        experiment_config.write_text(
            json.dumps(
                {
                    "protocol": "xception_frozen_square_224_aspect_balanced",
                    "manifest": summary["split_manifest"],
                    "image_size": summary["input_size"][0],
                    "crop_mode": summary["crop_mode"],
                    "aspect_balanced_sampling": summary[
                        "aspect_balanced_sampling"
                    ],
                    "selection_metric": "validation_macro_f1",
                    "test_evaluation": "disabled",
                    "test_images_loaded": False,
                    "test_predictions_saved": False,
                    "test_metrics_reported": False,
                    "experiment_reason": (
                        "Cấu hình được chọn bằng validation macro F1; "
                        "test bị loại khỏi vòng thử nghiệm."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    outputs = [
        run_path / "training_history.png",
        run_path / "validation_class_metrics.png",
        run_path / "validation_confusion_matrix_visual.png",
        run_path / "validation_aspect_analysis.png",
        run_path / "validation_analysis.md",
    ]
    plot_training_history(history, outputs[0], int(summary["best_epoch"]))
    plot_class_metrics(report, outputs[1])
    plot_confusion_matrix(
        run_path / "validation_confusion_matrix.csv", outputs[2]
    )
    plot_aspect_analysis(aspect_metrics, outputs[3])
    write_validation_analysis(summary, report, aspect_metrics, outputs[4])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Directory containing the Xception 224x224 result artifacts.",
    )
    args = parser.parse_args()
    for output in create_visual_report(args.run_dir):
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
