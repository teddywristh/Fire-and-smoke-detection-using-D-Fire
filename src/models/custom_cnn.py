"""Advanced Custom CNN: residual blocks with squeeze-and-excitation, trained from scratch.

Architecture and training settings come from notebooks/02_custom_cnn.ipynb
(the recorded v1 run). Data comes from the shared pipeline in src/datasets.py.

Run: python -m src.models.custom_cnn
Options: --evaluate-only (evaluate the saved checkpoint without training).
"""

import argparse
import json

# On Windows, import TensorFlow before numpy/sklearn/matplotlib.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers

import config
from src.datasets import INPUT_SHAPE, load_frames, make_dataset

MODEL_NAME = "Advanced Custom CNN"
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.05
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 12
MODEL_PATH = config.PROJECT_ROOT / "models" / "custom_cnn_best.keras"
RESULT_DIR = config.PROJECT_ROOT / "results" / "custom_cnn"
CRITICAL_ERRORS = [("fire", "nofire"), ("smokefire", "nofire")]


# Squeeze-and-Excitation: the network learns a weight for each channel.
def se_block(x, ratio=16):
    channels = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)                       # (H, W, C) -> (C,)
    s = layers.Dense(channels // ratio, activation="relu", use_bias=False)(s)
    s = layers.Dense(channels, activation="sigmoid", use_bias=False)(s)
    s = layers.Reshape((1, 1, channels))(s)
    return layers.Multiply()([x, s])


# Residual block + SE: output = F(x) + x
def residual_se_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)

    # 1x1 conv on the shortcut when the shape changes, so the two paths can be added.
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.Activation("swish")(x)


def build_model():
    inputs = layers.Input(shape=INPUT_SHAPE, name="input")
    x = layers.Rescaling(1 / 255, name="rescale")(inputs)       # shared pipeline gives [0, 255]

    # Stem: stride-2 conv, 224 -> 112
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)                          # (112, 112, 32)

    x = residual_se_block(x, filters=64, stride=1)             # (112, 112, 64)
    x = residual_se_block(x, filters=128, stride=2)            # (56, 56, 128)
    x = residual_se_block(x, filters=128, stride=1)
    x = residual_se_block(x, filters=256, stride=2)            # (28, 28, 256)
    x = residual_se_block(x, filters=256, stride=1)

    x = layers.GlobalAveragePooling2D(name="gap")(x)            # (28, 28, 256) -> (256,)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="swish")(x)
    outputs = layers.Dense(len(config.CLASS_NAMES), activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Advanced_Custom_CNN")
    # Label smoothing makes the model less over-confident on smoke vs smokefire.
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"])
    return model


def train(frames):
    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss",
                                           save_best_only=True, mode="min", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(RESULT_DIR / "training_log.csv")),
    ]
    model.fit(make_dataset(frames["train"], training=True, batch_size=BATCH_SIZE),
              validation_data=make_dataset(frames["val"], batch_size=BATCH_SIZE),
              epochs=EPOCHS, callbacks=callbacks, verbose=1)


def plot_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history["epoch"] + 1
    for ax, metric in [(ax1, "accuracy"), (ax2, "loss")]:
        ax.plot(epochs, history[metric], label=f"Train {metric}", color="steelblue", linewidth=2)
        ax.plot(epochs, history[f"val_{metric}"], label=f"Val {metric}", color="coral", linewidth=2)
        ax.set(title=metric.title(), xlabel="Epoch", ylabel=metric.title())
        ax.legend()
        ax.grid(True, alpha=0.4)
    fig.suptitle(f"{MODEL_NAME} – Learning Curves", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(matrix):
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, values, fmt, title in [(axes[0], matrix, "{:d}", "Counts"),
                                   (axes[1], normalized, "{:.2%}", "Row-normalized")]:
        fig.colorbar(ax.imshow(values, cmap="Blues"), ax=ax)
        ax.set(xticks=range(4), yticks=range(4), xticklabels=config.CLASS_NAMES,
               yticklabels=config.CLASS_NAMES, xlabel="Predicted", ylabel="True", title=title)
        for i, j in np.ndindex(values.shape):
            color = "white" if values[i, j] > values.max() / 2 else "black"
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", color=color)
    fig.suptitle(f"{MODEL_NAME} – Confusion Matrix", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_critical_errors(critical):
    shown = critical.head(16)
    columns = 4
    rows = max(1, int(np.ceil(len(shown) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, shown.itertuples()):
        ax.imshow(plt.imread(config.PROJECT_ROOT / row.processed_filepath))
        ax.set_title(f"True: {row.label}\nPred: {row.predicted_label} ({row.confidence:.2f})",
                     fontsize=9, color="darkred")
    if shown.empty:
        axes[0, 0].text(0.5, 0.5, "No critical errors", ha="center", va="center")
    fig.suptitle("Critical errors: fire / smokefire predicted as nofire", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "critical_errors.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def evaluate(model, frames):
    history = pd.read_csv(RESULT_DIR / "training_log.csv")
    test = frames["test"]
    dataset = make_dataset(test, batch_size=BATCH_SIZE)
    loss, _ = model.evaluate(dataset, verbose=0)
    probabilities = model.predict(dataset, verbose=0)
    predicted = probabilities.argmax(axis=1)
    labels = test["class_index"].to_numpy()

    report = classification_report(labels, predicted, labels=range(4), target_names=config.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.rename_axis("class").to_csv(RESULT_DIR / "classification_report.csv")
    matrix = confusion_matrix(labels, predicted, labels=range(4))
    pd.DataFrame(matrix, index=config.CLASS_NAMES, columns=config.CLASS_NAMES).rename_axis(
        "true_label").to_csv(RESULT_DIR / "confusion_matrix.csv")

    predictions = test[["processed_filepath", "label", "class_index"]].copy()
    predictions["predicted_index"] = predicted
    predictions["predicted_label"] = [config.CLASS_NAMES[i] for i in predicted]
    predictions["confidence"] = probabilities.max(axis=1)
    mistakes = predictions[predictions["class_index"] != predictions["predicted_index"]]
    mistakes.to_csv(RESULT_DIR / "misclassified.csv", index=False)
    is_critical = pd.Series(list(zip(mistakes["label"], mistakes["predicted_label"])),
                            index=mistakes.index).isin(CRITICAL_ERRORS)
    critical = mistakes[is_critical]

    metrics = {
        "model_name": MODEL_NAME,
        "test_accuracy": round(float(report["accuracy"]), 4),
        "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
        "test_macro_precision": round(report["macro avg"]["precision"], 4),
        "test_macro_recall": round(report["macro avg"]["recall"], 4),
        "test_loss": round(float(loss), 4),
        "critical_fire_as_nofire": int(matrix[config.CLASS_TO_INDEX["fire"], config.CLASS_TO_INDEX["nofire"]]),
        "critical_smokefire_as_nofire": int(
            matrix[config.CLASS_TO_INDEX["smokefire"], config.CLASS_TO_INDEX["nofire"]]),
        "misclassified_count": len(mistakes),
        "total_parameters": int(model.count_params()),
        "epochs_trained": len(history),
        "training_config": dict(learning_rate=LEARNING_RATE, label_smoothing=LABEL_SMOOTHING,
                                batch_size=BATCH_SIZE, epochs=EPOCHS, patience=PATIENCE,
                                seed=config.RANDOM_SEED,
                                data_pipeline="src.datasets (shared augmentation)"),
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")

    plot_learning_curves(history)
    plot_confusion_matrix(matrix)
    plot_critical_errors(critical)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    print(f"Train: {len(frames['train'])}, validation: {len(frames['val'])}, test: {len(frames['test'])}")

    if not args.evaluate_only:
        train(frames)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found; train first")
    model = tf.keras.models.load_model(MODEL_PATH)
    if not any(isinstance(layer, layers.Rescaling) for layer in model.layers):
        raise ValueError("Checkpoint predates the shared data pipeline (expects inputs in [0, 1]); "
                         "retrain with python -m src.models.custom_cnn")
    metrics = evaluate(model, frames)
    print(f"Done. Test accuracy: {metrics['test_accuracy']:.2%}; Macro F1: {metrics['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
