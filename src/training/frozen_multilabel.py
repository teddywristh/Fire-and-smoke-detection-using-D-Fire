# -*- coding: utf-8 -*-
"""
Thí nghiệm ResNet-50 đóng băng với đầu ra đa nhãn (fire, smoke).

Các class C4 mã hóa hai thuộc tính nhị phân: có lửa và có khói.
Thí nghiệm này học hai thuộc tính bằng BCE, rồi kết hợp lại thành 4 class C4.
Dữ liệu validation dùng để chọn hai ngưỡng quyết định; test chỉ được đánh giá
sau khi mô hình và ngưỡng đã cố định.

Chạy:
    python -m src.training.frozen_multilabel --epochs 20 --batch-size 32
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from src.training.frozen_transfer import (
    ManifestDataset,
    backbone_features,
    build_model,
    cache_features,
)


def attributes(labels):
    """
    Ánh xạ nhãn 4 class thành tensor [has_fire, has_smoke].
    """
    fire = (labels == config.CLASS_TO_INDEX["fire"]) | (
        labels == config.CLASS_TO_INDEX["smokefire"]
    )
    smoke = (labels == config.CLASS_TO_INDEX["smoke"]) | (
        labels == config.CLASS_TO_INDEX["smokefire"]
    )
    return torch.stack((fire, smoke), dim=1).float()

def combine_attributes(probabilities, fire_threshold, smoke_threshold):
    """
    Kết hợp xác suất [has_fire, has_smoke] thành dự đoán 4 class.
    """
    has_fire = probabilities[:, 0] >= fire_threshold
    has_smoke = probabilities[:, 1] >= smoke_threshold

    predictions = np.full(
        len(probabilities),
        config.CLASS_TO_INDEX["nofire"],
        dtype=np.int64,
    )
    predictions[has_fire & ~has_smoke] = config.CLASS_TO_INDEX["fire"]
    predictions[~has_fire & has_smoke] = config.CLASS_TO_INDEX["smoke"]
    predictions[has_fire & has_smoke] = config.CLASS_TO_INDEX["smokefire"]
    return predictions

def tune_thresholds(probabilities, labels):
    """
    Chọn ngưỡng fire/smoke để tối ưu macro F1 trên validation.
    """
    candidates = np.arange(0.25, 0.751, 0.025)

    best = (0.5, 0.5, -1.0)
    best_tie_distance = float("inf")

    for fire_threshold in candidates:
        for smoke_threshold in candidates:
            predictions = combine_attributes(
                probabilities, fire_threshold, smoke_threshold
            )
            score = f1_score(
                labels,
                predictions,
                labels=range(len(config.CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
            tie_distance = abs(fire_threshold - 0.5) + abs(smoke_threshold - 0.5)

            if score > best[2] or (
                score == best[2] and tie_distance < best_tie_distance
            ):
                best = (
                    float(fire_threshold),
                    float(smoke_threshold),
                    float(score),
                )
                best_tie_distance = tie_distance

    return best

@torch.no_grad()
def predict_probabilities(head, features, device, batch_size):
    """
    Chạy head trên tập features và trả về xác suất sigmoid.
    """
    head.eval()
    outputs = []

    for start in range(0, len(features), batch_size):
        logits = head(features[start:start + batch_size].to(device))
        outputs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(outputs)

def train_head_epoch(
    head,
    features,
    flipped,
    labels,
    criterion,
    optimizer,
    device,
    batch_size,
):
    """
    Huấn luyện head một epoch, dùng BCEWithLogitsLoss.
    """
    head.train()
    order = torch.randperm(len(labels))

    total_loss = 0.0
    correct = 0
    targets = attributes(labels)

    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch = features[indices].clone()

        choose_flip = torch.rand(len(indices)) < 0.5
        batch[choose_flip] = flipped[indices[choose_flip]]

        batch = batch.to(device)
        batch_targets = targets[indices].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = head(batch)
        loss = criterion(logits, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(indices)
        correct += (
            (logits.sigmoid().round() == batch_targets)
            .all(dim=1)
            .sum()
            .item()
        )

    return total_loss / len(labels), correct / len(labels)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = pd.read_csv(config.SPLIT_CSV_PATH)
    manifest = manifest[manifest.status == "ok"].copy()
    manifest["class_index"] = manifest.class_index.astype(int)

    missing = []
    for relative in manifest.processed_filepath:
        path = Path(relative)
        path = path if path.is_absolute() else config.PROJECT_ROOT / path
        if not path.is_file():
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "{} ảnh đã xử lý bị thiếu; hãy chạy "
            "`python -m src.data.data_loader` sau khi cấu hình DATASET_ROOT. "
            "File thiếu đầu tiên: {}".format(len(missing), missing[0])
        )
    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    datasets = {
        split: ManifestDataset(
            manifest[manifest.split == split], transform
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_model(
        "resnet50",
        2,
        pretrained=(not args.no_pretrained),
    ).to(device)

    head = model.fc
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    print(
        "Đang cache đặc trưng ResNet-50 cho train và validation...",
        flush=True,
    )
    train_features, flipped_features, train_labels = cache_features(
        model,
        "resnet50",
        loaders["train"],
        device,
        include_flipped=True,
    )
    val_features, _, val_labels = cache_features(
        model,
        "resnet50",
        loaders["val"],
        device,
    )
    val_labels_np = val_labels.numpy()

    output_dir = (
        config.PROJECT_ROOT
        / "results"
        / "transfer_frozen"
        / "resnet50_multilabel"
    )
    checkpoint_dir = (
        config.PROJECT_ROOT
        / "models"
        / "transfer_frozen"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "resnet50_multilabel_best.pt"

    history = []
    best_score = -1.0
    best_val_loss = float("inf")
    best_thresholds = (0.5, 0.5)

    for epoch in range(1, args.epochs + 1):
        train_loss, attribute_accuracy = train_head_epoch(
            head,
            train_features,
            flipped_features,
            train_labels,
            criterion,
            optimizer,
            device,
            args.batch_size,
        )

        probabilities = predict_probabilities(
            head, val_features, device, args.batch_size
        )

        fire_threshold, smoke_threshold, macro_f1 = tune_thresholds(
            probabilities, val_labels_np
        )

        val_predictions = combine_attributes(
            probabilities, fire_threshold, smoke_threshold
        )

        # Tính val_loss xấp xỉ từ probabilities
        eps = 1e-7
        probs_clipped = np.clip(probabilities, eps, 1 - eps)
        log_odds = np.log(probs_clipped / (1 - probs_clipped)).astype(np.float32)
        val_loss = criterion(
            torch.from_numpy(log_odds),
            attributes(val_labels),
        ).item()
        val_accuracy = float((val_predictions == val_labels_np).mean())
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_attribute_accuracy": attribute_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": macro_f1,
                "fire_threshold": fire_threshold,
                "smoke_threshold": smoke_threshold,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            "Epoch {:02d}/{:d} train_loss={:.4f} val_loss={:.4f} "
            "val_acc={:.4f} val_macro_f1={:.4f}".format(
                epoch,
                args.epochs,
                train_loss,
                val_loss,
                val_accuracy,
                macro_f1,
            ),
            flush=True,
        )

        if macro_f1 > best_score or (
            macro_f1 == best_score and val_loss < best_val_loss
        ):
            best_score = macro_f1
            best_val_loss = val_loss
            best_thresholds = (fire_threshold, smoke_threshold)

            torch.save(
                {
                    "architecture": "resnet50",
                    "task": "fire_smoke_attributes",
                    "class_names": config.CLASS_NAMES,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": macro_f1,
                    "val_loss": val_loss,
                    "thresholds": {
                        "fire": fire_threshold,
                        "smoke": smoke_threshold,
                    },
                    "seed": args.seed,
                },
                checkpoint_path,
            )
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )
    saved = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(saved["model_state_dict"])
    head = model.fc
    fire_threshold = saved["thresholds"]["fire"]
    smoke_threshold = saved["thresholds"]["smoke"]

    print("Đang cache đặc trưng cho tập test...", flush=True)
    test_features, _, test_labels = cache_features(
        model,
        "resnet50",
        loaders["test"],
        device,
    )
    test_probabilities = predict_probabilities(
        head, test_features, device, args.batch_size
    )

    y_true = test_labels.numpy()
    y_pred = combine_attributes(
        test_probabilities, fire_threshold, smoke_threshold
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(config.CLASS_NAMES))),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    with (output_dir / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(config.CLASS_NAMES))),
    )

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + config.CLASS_NAMES)
        for name, row in zip(config.CLASS_NAMES, matrix.tolist()):
            writer.writerow([name] + row)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")

    axis.set(
        xticks=range(len(config.CLASS_NAMES)),
        yticks=range(len(config.CLASS_NAMES)),
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Frozen ResNet-50 Multi-label - Test",
    )
    plt.setp(
        axis.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )

    max_val = matrix.max() if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > max_val / 2 else "black",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    print(
        "Checkpoint tốt nhất: {} (epoch {}, validation macro F1={:.4f})".format(
            str(checkpoint_path),
            saved["epoch"],
            saved["val_macro_f1"],
        )
    )
    print(
        "Test accuracy: {:.4f}; test macro F1: {:.4f}; "
        "kết quả chi tiết trong: {}".format(
            report["accuracy"],
            report["macro avg"]["f1-score"],
            str(output_dir),
        )
    )

if __name__ == "__main__":
    main()