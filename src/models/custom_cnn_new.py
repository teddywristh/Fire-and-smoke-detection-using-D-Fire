"""Custom CNN train từ scratch, dựa trên slide Chapter 5.2.

Đặt file vào src/models/custom_cnn_new.py trên branch revise/basic-complex.

Liên hệ với slide (số trang PDF):
  * Trang 5-8, VGG: lặp lại các Conv 3x3, stride 1, padding='same';
    sau mỗi khối dùng MaxPooling 2x2, stride 2.
  * Trang 18-22, Batch Normalization: dùng Conv -> BN -> ReLU.
  * Trang 9-12, NiN: mượn ý tưởng Global Average Pooling để giữ head nhỏ.
    Đây không phải NiN đầy đủ: không có mlpconv / chuỗi Conv 1x1.
  * Trang 3-4, AlexNet: dropout là một cách regularize head.

Điều chỉnh cho Forest Fire C4, không gọi là VGG-11/VGG-16 nguyên bản:
  * 4 khối, mỗi khối 2 Conv; filters = (32, 64, 128, 256).
    Giảm số filters để giảm tham số; 4 lần pooling cho map 14x14 thay vì
    map 7x7 của 5 lần pooling trên input 224x224.
  * GAP -> Dropout(0.3) -> Dense(4, softmax), thay head FC lớn của VGG.
    Dropout 0.3 là giá trị khởi đầu để kiểm chứng bằng validation.
  * 1,176,164 tham số khi có 4 lớp; số tham số không bảo đảm accuracy.

Dữ liệu dùng pipeline chung src.datasets hiện tại:
  RGB 224x224, float32 [0,255], one-hot labels; model chia 255 một lần.
  Augmentation chỉ do loader chung áp dụng cho train, không thêm ở đây.
  Adam, learning rate, label smoothing và callbacks giữ như CNN hiện tại.

Lệnh chạy từ thư mục gốc repo:
  python -m src.models.custom_cnn_new --summary-only
  python -m src.models.custom_cnn_new
  python -m src.models.custom_cnn_new --evaluate-only --split val
  python -m src.models.custom_cnn_new --evaluate-only --split test

Mặc định: train và đánh giá VAL. Chỉ chạy lệnh TEST sau khi chốt cấu hình.
Checkpoint: models/custom_cnn_new_best.keras
Kết quả: results/custom_cnn_new/ (test), results/custom_cnn_new/val/ (val).

Tài liệu gốc:
  VGG: https://arxiv.org/abs/1409.1556
  NiN: https://arxiv.org/abs/1312.4400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Giu thu tu import cua repo de tuong thich TensorFlow tren Windows.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers

import config
from src.datasets import INPUT_SHAPE, load_frames, make_dataset


MODEL_NAME = "Custom CNN New - Small VGG with BN and GAP"
FILTERS = (32, 64, 128, 256)
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.05
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 12
MODEL_PATH = config.PROJECT_ROOT / "models" / "custom_cnn_new_best.keras"
RESULT_DIR = config.PROJECT_ROOT / "results" / "custom_cnn_new"


def vgg_block(x, filters: int, block_id: int):
    """Slide 5-8: chuỗi Conv nhỏ, sau đó pooling; thêm BN từ slide 18-22."""
    for conv_id in (1, 2):
        prefix = f"block{block_id}_conv{conv_id}"
        x = layers.Conv2D(
            filters, kernel_size=3, strides=1, padding="same",
            use_bias=False, name=prefix,
        )(x)
        # BN da co beta (offset), nen Conv khong can them bias.
        x = layers.BatchNormalization(name=f"{prefix}_bn")(x)
        x = layers.Activation("relu", name=f"{prefix}_relu")(x)
    return layers.MaxPooling2D(
        pool_size=2, strides=2, name=f"block{block_id}_pool"
    )(x)


def build_model():
    """8 Conv + 1 Dense; tất cả weights được khởi tạo mới, không pretrained."""
    inputs = layers.Input(shape=INPUT_SHAPE, name="input")
    x = layers.Rescaling(1 / 255, name="rescale")(inputs)
    for block_id, filters in enumerate(FILTERS, start=1):
        x = vgg_block(x, filters, block_id)

    # Voi input 224: sau 4 khoi, feature map la (14, 14, 256).
    # GAP lay trung binh theo khong gian; khong flatten feature map thanh FC lon.
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(DROPOUT_RATE, name="head_dropout")(x)
    outputs = layers.Dense(
        len(config.CLASS_NAMES), activation="softmax", name="output"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="Custom_CNN_New")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=["accuracy"],
    )
    return model


def write_json(path: Path, payload: dict):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def plot_learning_curves(history: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history["epoch"] + 1, history[metric], label="Train")
        ax.plot(history["epoch"] + 1, history[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric, title=metric.title())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)


def train(frames: dict[str, pd.DataFrame], epochs: int = EPOCHS):
    """Chỉ train/val; checkpoint tốt nhất được chọn bằng val_loss."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (RESULT_DIR / "model_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    write_json(RESULT_DIR / "training_config.json", {
        "model_name": MODEL_NAME,
        "input_shape": list(INPUT_SHAPE),
        "class_names": list(config.CLASS_NAMES),
        "filters": list(FILTERS),
        "convs_per_block": 2,
        "dropout_rate": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "label_smoothing": LABEL_SMOOTHING,
        "batch_size": BATCH_SIZE,
        "max_epochs": epochs,
        "patience": PATIENCE,
        "seed": config.RANDOM_SEED,
        "checkpoint_monitor": "val_loss",
        "data_pipeline": "src.datasets (unchanged shared augmentation)",
        "tensorflow_version": tf.__version__,
    })
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss", mode="min",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(RESULT_DIR / "training_log.csv")),
    ]
    model.fit(
        make_dataset(frames["train"], training=True, batch_size=BATCH_SIZE),
        validation_data=make_dataset(frames["val"], batch_size=BATCH_SIZE),
        epochs=epochs, callbacks=callbacks, verbose=1,
    )
    plot_learning_curves(pd.read_csv(RESULT_DIR / "training_log.csv"))


def plot_confusion_matrix(matrix: np.ndarray, output_dir: Path, split: str):
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, values, fmt, title in (
        (axes[0], matrix, "{:d}", "Counts"),
        (axes[1], normalized, "{:.1%}", "Row-normalized"),
    ):
        fig.colorbar(ax.imshow(values, cmap="Blues"), ax=ax)
        n = len(config.CLASS_NAMES)
        ax.set(
            xticks=range(n), yticks=range(n),
            xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
            xlabel="Predicted", ylabel="True", title=f"{split}: {title}",
        )
        for i, j in np.ndindex(values.shape):
            color = "white" if values[i, j] > values.max() / 2 else "black"
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_errors(rows: pd.DataFrame, destination: Path, title: str):
    """Hiện ảnh raw nếu có; ghi rõ khi chỉ còn ảnh processed."""
    shown = rows.head(12)
    fig, axes = plt.subplots(
        max(1, (len(shown) + 3) // 4), 4,
        figsize=(12, 3.5 * max(1, (len(shown) + 3) // 4)), squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")
    if shown.empty:
        axes[0, 0].text(0.5, 0.5, "No errors", ha="center", va="center")
    for ax, row in zip(axes.flat, shown.itertuples()):
        raw_path = config.DATASET_ROOT / row.filepath
        image_path = raw_path if raw_path.exists() else config.PROJECT_ROOT / row.processed_filepath
        image_type = "Original" if raw_path.exists() else "Processed"
        with Image.open(image_path) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        ax.set_title(
            f"{image_type} | True: {row.label}\n"
            f"Pred: {row.predicted_label} ({row.confidence:.3f})", fontsize=9,
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def evaluate(model, frames: dict[str, pd.DataFrame], split: str = "val"):
    """Val dùng để so sánh; test được gọi riêng ở bước final evaluation."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    output_dir = RESULT_DIR / "val" if split == "val" else RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = frames[split]
    dataset = make_dataset(frame, batch_size=BATCH_SIZE)  # Khong shuffle/augment.
    loss = float(model.evaluate(dataset, verbose=0)[0])
    probabilities = model.predict(dataset, verbose=0)
    predicted = probabilities.argmax(axis=1)
    actual = frame["class_index"].to_numpy()
    n = len(config.CLASS_NAMES)
    report = classification_report(
        actual, predicted, labels=range(n), target_names=config.CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.rename_axis("class").to_csv(output_dir / "classification_report.csv")
    matrix = confusion_matrix(actual, predicted, labels=range(n))
    pd.DataFrame(matrix, index=config.CLASS_NAMES, columns=config.CLASS_NAMES).rename_axis(
        "true_label"
    ).to_csv(output_dir / "confusion_matrix.csv")

    predictions = frame[["filepath", "processed_filepath", "label", "class_index"]].copy()
    predictions["predicted_index"] = predicted
    predictions["predicted_label"] = [config.CLASS_NAMES[i] for i in predicted]
    predictions["confidence"] = probabilities.max(axis=1)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    mistakes = predictions[predictions["class_index"] != predictions["predicted_index"]]
    mistakes.to_csv(output_dir / "misclassified.csv", index=False)
    critical = mistakes[
        mistakes["label"].isin(["fire", "smokefire"])
        & (mistakes["predicted_label"] == "nofire")
    ]
    critical.to_csv(output_dir / "critical_errors.csv", index=False)

    # Moi cap nham lay toi da 2 anh, giup hien nhieu loai loi khac nhau.
    samples = mistakes.groupby(["label", "predicted_label"], sort=False).head(2)
    plot_errors(samples, output_dir / "misclassified_samples.png", f"{split}: error examples")
    plot_errors(critical, output_dir / "critical_errors.png", f"{split}: fire/smokefire -> nofire")
    plot_confusion_matrix(matrix, output_dir, split)

    history_path = RESULT_DIR / "training_log.csv"
    training_path = RESULT_DIR / "training_config.json"
    metrics = {
        "model_name": MODEL_NAME,
        "evaluation_split": split,
        f"{split}_accuracy": float(np.mean(actual == predicted)),
        f"{split}_macro_f1": float(report["macro avg"]["f1-score"]),
        f"{split}_macro_precision": float(report["macro avg"]["precision"]),
        f"{split}_macro_recall": float(report["macro avg"]["recall"]),
        f"{split}_loss": loss,
        "critical_fire_as_nofire": int(matrix[
            config.CLASS_TO_INDEX["fire"], config.CLASS_TO_INDEX["nofire"]
        ]),
        "critical_smokefire_as_nofire": int(matrix[
            config.CLASS_TO_INDEX["smokefire"], config.CLASS_TO_INDEX["nofire"]
        ]),
        "misclassified_count": len(mistakes),
        "total_parameters": int(model.count_params()),
        "epochs_trained": len(pd.read_csv(history_path)) if history_path.exists() else None,
        "training_config": json.loads(training_path.read_text(encoding="utf-8"))
            if training_path.exists() else None,
    }
    write_json(output_dir / "metrics.json", metrics)
    print(
        f"{split}: accuracy={metrics[f'{split}_accuracy']:.2%}; "
        f"macro F1={metrics[f'{split}_macro_f1']:.4f}; errors={len(mistakes)}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary-only", action="store_true", help="Xem kien truc, khong can dataset")
    mode.add_argument("--evaluate-only", action="store_true", help="Danh gia checkpoint da luu")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.split == "test" and not args.evaluate_only:
        parser.error("Final test: use --evaluate-only --split test after selecting the model")

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    if args.summary_only:
        build_model().summary()
        return
    if args.evaluate_only and not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found; train first")
    frames = load_frames()
    if not args.evaluate_only:
        train(frames, epochs=args.epochs)
    model = tf.keras.models.load_model(MODEL_PATH)
    evaluate(model, frames, split=args.split)


if __name__ == "__main__":
    main()
