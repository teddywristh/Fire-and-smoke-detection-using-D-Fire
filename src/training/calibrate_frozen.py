# -*- coding: utf-8 -*-
"""
Hiệu chỉnh (calibrate) logits của mô hình ResNet-50 đã được giữ cố định,
chỉ dùng dữ liệu validation.

- Checkpoint tốt nhất ban đầu được giữ nguyên, không sửa.
- Dùng macro F1 trên tập validation để tìm các bias cộng thêm cho từng class.
- Sau khi chọn được bias, mô hình đã hiệu chỉnh được cố định và đánh giá trên test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.training.frozen_transfer import ManifestDataset, build_model

@torch.no_grad()
def collect_logits(model, loader, device):
    """
    model: mô hình PyTorch đã được huấn luyện.
    loader: DataLoader chứa ảnh và nhãn.
    device: thiết bị tính toán (cuda/cpu).

        - logits_all: mảng numpy chứa logits cho tất cả mẫu.
        - labels_all: mảng numpy chứa nhãn thật cho tất cả mẫu.
    """
    logits_all = []
    labels_all = []
    model.eval()

    for images, labels in loader:
        logits_batch = model(images.to(device)).cpu().numpy()
        labels_batch = labels.numpy()
        logits_all.append(logits_batch)
        labels_all.append(labels_batch)

    logits_all = np.concatenate(logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return logits_all, labels_all

def tune_biases(logits, labels):
    """
    logits: mảng numpy (num_samples, num_classes) chứa logits gốc.
    labels: mảng numpy (num_samples,) chứa nhãn thật.
        - biases: mảng numpy (num_classes,) chứa bias cho từng class.
        - best_score[0]: macro F1 tốt nhất đạt được trên validation.
    """
    num_classes = logits.shape[1]
    biases = np.zeros(num_classes, dtype=np.float32)

    def score(values):
        """
        Tính macro F1 và accuracy khi cộng thêm bias vào logits.
        """
        predictions = (logits + values).argmax(axis=1)
        macro_f1 = f1_score(
            labels, predictions, average="macro", zero_division=0
        )
        accuracy = float((predictions == labels).mean())
        return macro_f1, accuracy
    best_score = score(biases)

    search_config = [
        (0.2, 2.0, 3),   # (step, radius, passes)
        (0.05, 0.4, 2),
        (0.01, 0.1, 2),
    ]

    for step, radius, passes in search_config:
        offsets = np.arange(-radius, radius + step / 2, step)
        for _ in range(passes):
            changed = False
            for class_index in range(num_classes):
                candidates = []
                for offset in offsets:
                    candidate = biases.copy()
                    candidate[class_index] += offset
                    candidate_score = score(candidate)
                    candidates.append((candidate_score, candidate))
                candidate_score, candidate_bias = max(
                    candidates,
                    key=lambda item: (
                        item[0][0],           # macro F1
                        item[0][1],           # accuracy
                        -float(np.abs(item[1]).sum()),  # tổng |bias| nhỏ
                    ),
                )
                if candidate_score > best_score:
                    biases = candidate_bias
                    best_score = candidate_score
                    changed = True

            if not changed:
                #early stop
                break

    return biases, best_score[0]

def main():
    """
      - Đọc manifest, tạo DataLoader cho val và test.
      - Nạp checkpoint mô hình.
      - Thu thập logits trên val và test.
      - Tìm bias tối ưu macro F1 trên val.
      - Áp dụng bias vào mô hình.
      - Đánh giá lại trên test và ghi kết quả.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models/transfer_frozen/resnet50_best.pt",
        help="Đường dẫn tới checkpoint mô hình cần hiệu chỉnh",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Kích thước batch khi chạy inference",
    )
    args = parser.parse_args()
    manifest = pd.read_csv(config.SPLIT_CSV_PATH)
    ok_mask = manifest.status == "ok"
    manifest = manifest[ok_mask].copy()
    manifest["class_index"] = manifest.class_index.astype(int)

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    loaders = {}
    for split in ("val", "test"):
        sub_manifest = manifest[manifest.split == split]
        dataset = ManifestDataset(sub_manifest, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        loaders[split] = loader

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        "resnet50",
        len(config.CLASS_NAMES),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_logits, val_labels = collect_logits(model, loaders["val"], device)
    baseline_val_f1 = f1_score(
        val_labels,
        val_logits.argmax(axis=1),
        average="macro",
        zero_division=0,
    )
    biases, tuned_val_f1 = tune_biases(val_logits, val_labels)
    bias_tensor = torch.as_tensor(
        biases,
        device=device,
        dtype=model.fc.bias.dtype,
    )
    model.fc.bias.data.add_(bias_tensor)
    test_logits, test_labels = collect_logits(model, loaders["test"], device)
    test_predictions = test_logits.argmax(axis=1)
    report = classification_report(
        test_labels,
        test_predictions,
        labels=list(range(len(config.CLASS_NAMES))),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    baseline_test_logits = test_logits - biases
    baseline_test_predictions = baseline_test_logits.argmax(axis=1)
    baseline_test_report = classification_report(
        test_labels,
        baseline_test_predictions,
        labels=list(range(len(config.CLASS_NAMES))),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    output_dir = PROJECT_ROOT / "results/transfer_frozen/resnet50_calibrated"
    checkpoint_dir = PROJECT_ROOT / "models/transfer_frozen"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    calibrated_checkpoint_path = checkpoint_dir / "resnet50_calibrated.pt"
    calibrated_checkpoint = {
        **checkpoint,
        "model_state_dict": model.state_dict(),
        "calibration_biases": biases.tolist(),
        "val_macro_f1": tuned_val_f1,
    }
    torch.save(calibrated_checkpoint, calibrated_checkpoint_path)
    comparison_path = output_dir / "comparison.json"
    comparison_data = {
        "validation_macro_f1_before": baseline_val_f1,
        "validation_macro_f1_after": tuned_val_f1,
        "biases": dict(zip(config.CLASS_NAMES, biases.tolist())),
        "test_before": baseline_test_report,
        "test_after": report,
    }
    with comparison_path.open("w", encoding="utf-8") as file:
        json.dump(comparison_data, file, indent=2)
    print(
        "Validation macro F1: {:.4f} -> {:.4f}".format(
            baseline_val_f1, tuned_val_f1
        )
    )
    print(
        "Test accuracy: {:.4f} -> {:.4f}".format(
            baseline_test_report["accuracy"],
            report["accuracy"],
        )
    )
    print(
        "Test macro F1: {:.4f} -> {:.4f}".format(
            baseline_test_report["macro avg"]["f1-score"],
            report["macro avg"]["f1-score"],
        )
    )
    print(
        "Biases ({}): {}".format(
            ", ".join(config.CLASS_NAMES),
            biases.tolist(),
        )
    )
    print("File so sánh: {}".format(str(comparison_path)))
    print(
        "Checkpoint đã hiệu chỉnh: {}".format(str(calibrated_checkpoint_path))
    )

if __name__ == "__main__":
    main()