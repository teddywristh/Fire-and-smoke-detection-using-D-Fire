"""Training, evaluation and reporting shared by the four notebooks.

Every run is selected on validation only. Test is predicted once, after the
best epoch is reloaded, and reported next to validation.
"""

import json
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config

CLASSES = config.CLASS_NAMES

# name: (constructor, input scaling the ImageNet weights expect, first layer of the last stage)
BACKBONES = {
    "xception": (keras.applications.Xception,
                 lambda: keras.layers.Rescaling(1 / 127.5, offset=-1, name="to_minus1_1"),
                 "block13"),
    "resnet50": (keras.applications.ResNet50,
                 lambda: keras.layers.Lambda(keras.applications.resnet50.preprocess_input, name="caffe_bgr"),
                 "conv5_"),
}


def setup(seed=config.RANDOM_SEED, mixed_precision=False):
    """Call first in every notebook: GPU memory growth, precision policy, seeds."""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # GPU already initialised in this kernel
    keras.mixed_precision.set_global_policy("mixed_float16" if mixed_precision else "float32")
    keras.utils.set_random_seed(seed)
    print("GPU:", tf.config.list_physical_devices("GPU") or "none", "| policy:", keras.mixed_precision.global_policy().name)


def backbone(name):
    """Frozen ImageNet backbone with global average pooling, and its input-scaling layer."""
    constructor, preprocess, _ = BACKBONES[name]
    base = constructor(include_top=False, weights="imagenet",
                       input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3), pooling="avg")
    base.trainable = False
    return base, preprocess()


def unfreeze(base, name, scope):
    """scope="top": last stage only (Xception exit flow, ResNet50 conv5); "all": every layer.

    BatchNormalization layers stay frozen in both cases.
    """
    start = 0
    if scope == "top":
        start = next(i for i, layer in enumerate(base.layers) if layer.name.startswith(BACKBONES[name][2]))
    base.trainable = True
    for index, layer in enumerate(base.layers):
        layer.trainable = index >= start and not isinstance(layer, keras.layers.BatchNormalization)


def compile_model(model, optimizer, loss="categorical_crossentropy"):
    # No XLA: it asks for one 3.9 GB buffer when the whole Xception is unfrozen (6 GB GPU).
    model.compile(optimizer=optimizer, loss=loss, jit_compile=False,
                  metrics=["accuracy", keras.metrics.F1Score(average="macro", name="f1")])


def count_params(model):
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    return model.count_params(), trainable


def fit(model, run, datasets, epochs, monitor="val_loss", early_stopping=None, reduce_lr=None):
    """Train, keep the best epoch by `monitor` and reload it.

    early_stopping: patience on `monitor`, or None.
    reduce_lr: dict(factor, patience, min_lr) for ReduceLROnPlateau on val_loss, or None.
    """
    mode = "max" if monitor == "val_f1" else "min"
    run_dir = config.RESULTS_DIR / run
    run_dir.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = config.MODELS_DIR / f"{run}.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint, monitor=monitor, mode=mode,
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.CSVLogger(run_dir / "history.csv"),
    ]
    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=early_stopping))
    if reduce_lr:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", **reduce_lr))
    started = time.time()
    history = model.fit(datasets["train"], validation_data=datasets["val"], epochs=epochs,
                        callbacks=callbacks, shuffle=False, verbose=2)  # the dataset samples itself
    minutes = (time.time() - started) / 60
    model.load_weights(checkpoint)
    history = pd.DataFrame(history.history, index=pd.RangeIndex(1, len(history.history["loss"]) + 1, name="epoch"))
    best = history[monitor].idxmax() if mode == "max" else history[monitor].idxmin()
    return {"history": history, "monitor": monitor, "best_epoch": int(best),
            "epochs_run": len(history), "train_minutes": round(minutes, 1)}


def _split_metrics(model, frame, dataset):
    probabilities = model.predict(dataset, verbose=0)
    truth, predicted = frame.class_index.to_numpy(), probabilities.argmax(1)
    report = classification_report(truth, predicted, labels=range(len(CLASSES)), target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=range(len(CLASSES)))
    predictions = frame[["filepath", "label", "aspect_group"]].assign(
        predicted=[CLASSES[i] for i in predicted], confidence=probabilities.max(1).round(4))
    by_aspect = (predictions.assign(correct=predictions.label == predictions.predicted)
                 .groupby(["label", "aspect_group"]).correct.agg(["mean", "size"]))
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score")} for c in CLASSES},
        "confusion_matrix": matrix.tolist(),
        "recall_by_label_and_aspect": {f"{label}/{group}": {"recall": round(row["mean"], 4), "n": int(row["size"])}
                                       for (label, group), row in by_aspect.iterrows()},
    }
    return metrics, predictions, matrix


def evaluate(model, run, task, frames, datasets, fit_info, settings):
    """Score val and test, write results/<run>/ and update results/leaderboard.csv."""
    run_dir = config.RESULTS_DIR / run
    run_dir.mkdir(parents=True, exist_ok=True)
    total, trainable = count_params(model)
    report = {"task": task, "run": run, "settings": settings, "total_params": total,
              "trainable_params": trainable, **{k: v for k, v in fit_info.items() if k != "history"}}
    matrices = {}
    for split in ("val", "test"):
        report[split], predictions, matrices[split] = _split_metrics(model, frames[split], datasets[split])
        predictions.to_csv(run_dir / f"{split}_predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot_run(run, fit_info, matrices, run_dir / "summary.png")
    row = _leaderboard_row(report, matrices["test"])
    update_leaderboard(row)
    return pd.Series(row)


def _leaderboard_row(report, test_matrix):
    index = config.CLASS_TO_INDEX
    test = report["test"]
    row = {
        "task": report["task"], "run": report["run"],
        "trainable_params": report["trainable_params"], "total_params": report["total_params"],
        "best_epoch": report["best_epoch"], "epochs_run": report["epochs_run"],
        "train_minutes": report["train_minutes"],
        "val_acc": report["val"]["accuracy"], "val_f1": report["val"]["macro_f1"],
        "test_acc": test["accuracy"], "test_f1": test["macro_f1"],
    }
    row.update({f"test_f1_{c}": test["per_class"][c]["f1-score"] for c in CLASSES})
    row["test_smokefire_as_smoke"] = int(test_matrix[index["smokefire"], index["smoke"]])
    row["test_fire_as_smokefire"] = int(test_matrix[index["fire"], index["smokefire"]])
    row["test_fire_or_smokefire_as_nofire"] = int(test_matrix[index["fire"], index["nofire"]]
                                                  + test_matrix[index["smokefire"], index["nofire"]])
    return row


def update_leaderboard(row):
    """One row per run in results/leaderboard.csv (re-running a run replaces its row)."""
    path = config.RESULTS_DIR / "leaderboard.csv"
    board = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not board.empty:
        board = board[board.run != row["run"]]
    board = pd.concat([board, pd.DataFrame([row])], ignore_index=True)
    board.to_csv(path, index=False, float_format="%.4f")
    return board


def _plot_run(run, fit_info, matrices, path):
    history = fit_info["history"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, metric in zip(axes[:2], ("loss", "f1")):
        ax.plot(history.index, history[metric], marker=".", label="train")
        ax.plot(history.index, history[f"val_{metric}"], marker=".", label="val")
        ax.axvline(fit_info["best_epoch"], color="grey", linestyle=":", label="best epoch")
        ax.set(title=metric if metric == "loss" else "macro F1", xlabel="epoch")
        ax.legend()
    for ax, split in zip(axes[2:], ("val", "test")):
        matrix = matrices[split]
        ax.imshow(matrix, cmap="Blues")
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xticklabels=CLASSES,
               yticklabels=CLASSES, xlabel="predicted", ylabel="true", title=f"{split} confusion")
    fig.suptitle(run)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.show()


def preview(dataset, rows=2, cols=8):
    """Show one batch the way the models receive it."""
    images, labels = next(iter(dataset))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.1 * rows))
    for ax, image, label in zip(axes.flat, images, labels):
        ax.imshow(image.numpy().astype("uint8"))
        ax.set_title(CLASSES[int(np.argmax(label))], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    plt.show()
