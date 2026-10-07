# -*- coding: utf-8 -*-
"""
Transfer learning với Xception đóng băng, đầu vào được letterbox để giữ nguyên aspect.
Ảnh thô được letterbox về kích thước đầu vào của Xception pretrained.
Đặc trưng frozen được cache trên disk theo từng split, sau đó huấn luyện
một classifier nhỏ. Không sử dụng hay sửa đổi preprocessing và checkpoint
của ResNet.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config

CACHE_VERSION = "xception-crop-protocol-v2"
SQUARE_RANGE = (0.9, 1.1)
WIDE_RANGE = (1.6, 1.95)

def aspect_group(ratio):
    if SQUARE_RANGE[0] <= ratio <= SQUARE_RANGE[1]:
        return "square"
    if WIDE_RANGE[0] <= ratio <= WIDE_RANGE[1]:
        return "wide"
    return "other"

def letterbox(image, size):
    image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(
        image,
        ((size - image.width) // 2, (size - image.height) // 2),
    )
    return canvas

def augment_image(image):
    top, left, height, width = RandomResizedCrop.get_params(
        image, scale=(0.72, 1.0), ratio=(0.75, 1.9)
    )
    image = image.crop((left, top, left + width, top + height))
    out_w, out_h = image.size

    image = TF.affine(
        image,
        angle=random.uniform(-5.0, 5.0),
        translate=(
            round(out_w * random.uniform(-0.04, 0.04)),
            round(out_h * random.uniform(-0.04, 0.04)),
        ),
        scale=random.uniform(0.94, 1.06),
        shear=(
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
        ),
        interpolation=InterpolationMode.BILINEAR,
        fill=(0, 0, 0),
    )

    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    image = ImageEnhance.Brightness(image).enhance(
        random.uniform(0.88, 1.12)
    )
    image = ImageEnhance.Contrast(image).enhance(
        random.uniform(0.88, 1.12)
    )
    return image

def square_crop(image, training):
    width, height = image.size
    side = min(width, height)
    if training:
        side = max(1, round(side * random.uniform(0.72, 1.0)))
        left = (
            random.randint(0, width - side)
            if width > side
            else 0
        )
        top = (
            random.randint(0, height - side)
            if height > side
            else 0
        )
    else:
        left, top = (width - side) // 2, (height - side) // 2

    return image.crop((left, top, left + side, top + side))

def augment_square_image(image):
    width, height = image.size
    image = TF.affine(
        image,
        angle=random.uniform(-5.0, 5.0),
        translate=(
            round(width * random.uniform(-0.04, 0.04)),
            round(height * random.uniform(-0.04, 0.04)),
        ),
        scale=random.uniform(0.94, 1.06),
        shear=(
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
        ),
        interpolation=InterpolationMode.BILINEAR,
        fill=(0, 0, 0),
    )
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(
        random.uniform(0.88, 1.12)
    )
    return ImageEnhance.Contrast(image).enhance(
        random.uniform(0.88, 1.12)
    )

class RawManifestDataset(Dataset):
    def __init__(self, frame, size, training, crop_mode="letterbox"):
        self.frame = frame.reset_index(drop=True)
        self.size = size
        self.training = training
        self.crop_mode = crop_mode
        self.to_tensor = lambda image: torch.from_numpy(
            np.asarray(image, dtype=np.float32)
            .transpose(2, 0, 1)
            .copy()
            / 127.5
            - 1.0
        )
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row.filepath)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        if self.crop_mode == "square":
            image = square_crop(image, training=self.training)
            image = image.resize(
                (self.size, self.size), Image.Resampling.BILINEAR
            )
            if self.training:
                image = augment_square_image(image)
        else:
            if self.training:
                image = augment_image(image)
            image = letterbox(image, self.size)

        ratio = float(row.width) / float(row.height)
        return self.to_tensor(image), int(row.class_index), ratio

def cache_fingerprint(frame, split, weights_id, input_size, crop_mode):
    rows = frame[["filepath", "raw_sha256", "class_index", "width", "height"]]
    size_key = "" if input_size == 299 else str(input_size)

    payload = (
        CACHE_VERSION
        + size_key
        + weights_id
        + split
        + crop_mode
        + rows.to_json(orient="records")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

@torch.inference_mode()
def cache_features(
    model, loader, device, cache_path, fingerprint, split
):
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            if str(archive["fingerprint"].item()) == fingerprint:
                print(
                    "Dùng cache {} features: {}".format(split, cache_path),
                    flush=True,
                )
                return {
                    key: archive[key]
                    for key in ("features", "labels", "ratios")
                }

        print(
            "Cache {} không hợp lệ; đang rebuild.".format(split),
            flush=True,
        )
    feature_parts = []
    label_parts = []
    ratio_parts = []
    model.eval()
    total_batches = len(loader)

    for batch_index, (images, labels, ratios) in enumerate(
        loader, start=1
    ):
        features = model(images.to(device)).float().cpu().numpy()
        feature_parts.append(features)
        label_parts.append(labels.numpy())
        ratio_parts.append(ratios.numpy())
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
            print(
                "Cache {}: batch {}/{}".format(split, batch_index, total_batches),
                flush=True,
            )
    arrays = {
        "features": np.concatenate(feature_parts).astype(
            np.float32, copy=False
        ),
        "labels": np.concatenate(label_parts).astype(np.int64, copy=False),
        "ratios": np.concatenate(ratio_parts).astype(np.float32, copy=False),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_path, **arrays, fingerprint=np.asarray(fingerprint)
    )
    temp_path.replace(cache_path)

    return arrays

def audit_aspects(manifest, output_dir):
    frame = manifest.copy()
    frame["aspect_ratio"] = frame.width.astype(float) / frame.height.astype(float)
    frame["aspect_group"] = frame.aspect_ratio.map(aspect_group)

    audit = {}
    for split, split_frame in frame.groupby("split"):
        audit[split] = {}
        for label, class_frame in split_frame.groupby("label"):
            counts = class_frame.aspect_group.value_counts()
            audit[split][label] = {
                "count": int(len(class_frame)),
                "median_aspect_ratio": float(
                    class_frame.aspect_ratio.median()
                ),
                "square_count": int(counts.get("square", 0)),
                "wide_count": int(counts.get("wide", 0)),
                "other_count": int(counts.get("other", 0)),
            }

    with (output_dir / "aspect_audit.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(audit, file, indent=2)

    return audit

def train_epoch(
    head,
    features,
    labels,
    criterion,
    optimizer,
    device,
    batch_size,
    sample_weights=None,
):
    head.train()
    if sample_weights is not None:
        order = torch.multinomial(
            sample_weights, len(labels), replacement=True
        )
    else:
        order = torch.randperm(len(labels))
    loss_sum = 0.0
    correct = 0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        if len(indices) == 1:
            continue
        x = features[indices].to(device)
        y = labels[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = head(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(indices)
        correct += (logits.argmax(1) == y).sum().item()
    return loss_sum / len(labels), correct / len(labels)

@torch.inference_mode()
def evaluate_head(head, features, labels, criterion, device, batch_size):
    head.eval()
    logits_parts = []
    loss_sum = 0.0
    correct = 0

    for start in range(0, len(labels), batch_size):
        x = features[start:start + batch_size].to(device)
        y = labels[start:start + batch_size].to(device)

        logits = head(x)
        loss_sum += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        logits_parts.append(logits.cpu())

    logits = torch.cat(logits_parts)
    return (
        loss_sum / len(labels),
        correct / len(labels),
        logits.argmax(1).numpy(),
    )

def save_confusion(matrix, path, title):
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(config.CLASS_NAMES)),
        yticks=range(len(config.CLASS_NAMES)),
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(
        axis.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--image-size",
        type=int,
        choices=[64, 224, 256, 299, 320, 384],
        default=299,
        help="Kích thước letterbox vuông; mặc định pretrained là 299.",
    )
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--split-csv", type=Path, default=config.SPLIT_CSV_PATH)
    parser.add_argument(
        "--crop-mode",
        choices=["letterbox", "square"],
        default="letterbox",
    )
    parser.add_argument("--aspect-balance", action="store_true")
    args = parser.parse_args()

    if args.cpu_threads < 1 or args.batch_size < 2:
        parser.error(
            "--cpu-threads phải >= 1 và --batch-size phải >= 2"
        )

    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    new_protocol = (
        args.crop_mode == "square"
        or args.aspect_balance
        or args.split_csv.resolve() != config.SPLIT_CSV_PATH.resolve()
    )

    if new_protocol and args.aspect_balance:
        run_name = "xception_{}_aspect_balanced_{}".format(
            args.crop_mode, args.image_size
        )
    elif new_protocol:
        run_name = "xception_{}_{}".format(args.crop_mode, args.image_size)
    elif args.image_size == 299:
        run_name = "xception_aspect_robust"
    else:
        run_name = "xception_aspect_robust_{}".format(args.image_size)

    output_dir = PROJECT_ROOT / "results/transfer_frozen" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.split_csv)
    required = {
        "filepath",
        "raw_sha256",
        "split",
        "label",
        "class_index",
        "width",
        "height",
        "status",
    }
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            "split.csv thiếu các cột: {}".format(sorted(missing_columns))
        )
    manifest = manifest[manifest.status == "ok"].copy()
    manifest["class_index"] = manifest.class_index.astype(int)
    if any(
        manifest[manifest.split == split].empty
        for split in ("train", "val")
    ):
        raise ValueError(
            "Phải có hàng train và val trong file split CSV đã chọn"
        )
    manifest = manifest[manifest.split.isin(["train", "val"])].copy()
    missing_paths = []
    for relative in manifest.filepath:
        path = Path(relative)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            missing_paths.append(path)

    if missing_paths:
        raise FileNotFoundError(
            "{} ảnh thô bị thiếu; ví dụ: {}".format(
                len(missing_paths), missing_paths[0]
            )
        )
    audit = audit_aspects(manifest, output_dir)
    print("Kiểm tra aspect-ratio (số lượng / median / square / wide):", flush=True)
    for split, classes in audit.items():
        for label, values in classes.items():
            print(
                "  {:5s} {:9s} {:4d} / {:.3f} / {} / {}".format(
                    split,
                    label,
                    values["count"],
                    values["median_aspect_ratio"],
                    values["square_count"],
                    values["wide_count"],
                ),
                flush=True,
            )
    model = timm.create_model(
        "legacy_xception", pretrained=True, num_classes=0
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    weights_id = str(
        model.pretrained_cfg.get("url", "xception-imagenet")
    )
    pretrained_size = int(model.pretrained_cfg["input_size"][-1])
    input_size = args.image_size or pretrained_size
    mean = (
        torch.tensor(model.pretrained_cfg["mean"], dtype=torch.float32)
        .view(1, 3, 1, 1)
    )
    std = (
        torch.tensor(model.pretrained_cfg["std"], dtype=torch.float32)
        .view(1, 3, 1, 1)
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    class NormalizeInput(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            image, label, ratio = self.dataset[index]
            return image, label, ratio

    split_tag = hashlib.sha256(
        str(args.split_csv.resolve()).encode("utf-8")
    ).hexdigest()[:8]

    cache_dir = (
        PROJECT_ROOT
        / "data/cache"
        / "{}_{}_v2".format(run_name, split_tag)
    )
    cached = {}
    loaders = {}
    for split in ("train", "val"):
        frame = manifest[manifest.split == split]
        dataset = RawManifestDataset(
            frame,
            input_size,
            training=(split == "train"),
            crop_mode=args.crop_mode,
        )
        wrapped = NormalizeInput(dataset)

        loaders[split] = DataLoader(
            wrapped,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        cache_path = cache_dir / "{}.npz".format(split)
        fingerprint = cache_fingerprint(
            frame, split, weights_id, input_size, args.crop_mode
        )

        cached[split] = cache_features(
            model, loaders[split], device, cache_path, fingerprint, split
        )

    del loaders
    feature_dim = int(cached["train"]["features"].shape[1])
    head = nn.Sequential(
        nn.Dropout(0.45),
        nn.Linear(feature_dim, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, len(config.CLASS_NAMES)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    train_x = torch.from_numpy(cached["train"]["features"])
    train_y = torch.from_numpy(cached["train"]["labels"])
    val_x = torch.from_numpy(cached["val"]["features"])
    val_y = torch.from_numpy(cached["val"]["labels"])
    sample_weights = None
    if args.aspect_balance:
        groups = [
            aspect_group(float(r)) for r in cached["train"]["ratios"]
        ]
        labels = cached["train"]["labels"]
        keys = [
            (int(y), group) for y, group in zip(labels, groups)
        ]
        counts = pd.Series(keys).value_counts().to_dict()

        weights = np.asarray(
            [1.0 / np.sqrt(counts[key]) for key in keys],
            dtype=np.float64,
        )
        sample_weights = torch.as_tensor(
            weights / weights.sum(), dtype=torch.float32
        )
        print(
            "Số lượng class/group (aspect-aware): {}".format(counts),
            flush=True,
        )
    checkpoint = (
        PROJECT_ROOT
        / "models/transfer_frozen"
        / "{}_best.pt".format(run_name)
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -1.0
    best_val_loss = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            head,
            train_x,
            train_y,
            criterion,
            optimizer,
            device,
            args.batch_size,
            sample_weights=sample_weights,
        )
        val_loss, val_accuracy, val_pred = evaluate_head(
            head, val_x, val_y, criterion, device, args.batch_size
        )
        val_f1 = float(
            f1_score(
                val_y.numpy(),
                val_pred,
                average="macro",
                zero_division=0,
            )
        )
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            "Epoch {:02d}/{:d} train_acc={:.4f} val_loss={:.4f} val_f1={:.4f}".format(
                epoch, args.epochs, train_accuracy, val_loss, val_f1
            ),
            flush=True,
        )

        if val_f1 > best_f1 or (
            val_f1 == best_f1 and val_loss < best_val_loss
        ):
            best_f1 = val_f1
            best_val_loss = val_loss
            stale = 0
            torch.save(
                {
                    "architecture": "legacy_xception",
                    "class_names": config.CLASS_NAMES,
                    "backbone_state_dict": model.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                    "cache_version": CACHE_VERSION,
                    "seed": args.seed,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(
                    "Dừng sớm sau {} epoch.".format(epoch),
                    flush=True,
                )
                break

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )
    saved = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(saved["backbone_state_dict"])
    head.load_state_dict(saved["head_state_dict"])

    _, val_accuracy, val_pred = evaluate_head(
        head, val_x, val_y, criterion, device, args.batch_size
    )
    val_true = val_y.numpy()
    val_report = classification_report(
        val_true,
        val_pred,
        labels=list(range(len(config.CLASS_NAMES))),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    with (output_dir / "validation_classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(val_report, file, indent=2)
    val_matrix = confusion_matrix(
        val_true,
        val_pred,
        labels=list(range(len(config.CLASS_NAMES))),
    )
    with (output_dir / "validation_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + config.CLASS_NAMES)
        for label, row in zip(config.CLASS_NAMES, val_matrix.tolist()):
            writer.writerow([label] + row)
    save_confusion(
        val_matrix,
        output_dir / "validation_confusion_matrix.png",
        "Xception Validation",
    )
    val_ratios = cached["val"]["ratios"]
    val_group_names = np.asarray(
        [aspect_group(float(r)) for r in val_ratios]
    )
    val_groups = {}
    for group in ("square", "wide", "other"):
        mask = val_group_names == group
        if not mask.any():
            continue

        group_true = val_true[mask]
        group_pred = val_pred[mask]

        group_report = classification_report(
            group_true,
            group_pred,
            labels=list(range(len(config.CLASS_NAMES))),
            target_names=config.CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        val_groups[group] = {
            "support": int(mask.sum()),
            "class_support": {
                name: int((group_true == i).sum())
                for i, name in enumerate(config.CLASS_NAMES)
            },
            "accuracy": float((group_true == group_pred).mean()),
            "macro_f1_supported_classes": float(
                f1_score(
                    group_true,
                    group_pred,
                    labels=sorted(np.unique(group_true).tolist()),
                    average="macro",
                    zero_division=0,
                )
            ),
            "class_report": group_report,
        }
    with (output_dir / "validation_aspect_group_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(val_groups, file, indent=2)
    summary = {
        "architecture": "legacy_xception",
        "input_size": [input_size, input_size],
        "normalization_mean": model.pretrained_cfg["mean"],
        "normalization_std": model.pretrained_cfg["std"],
        "best_epoch": int(saved["epoch"]),
        "best_validation_macro_f1": float(saved["val_macro_f1"]),
        "best_validation_loss": float(saved["val_loss"]),
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": float(
            val_report["macro avg"]["f1-score"]
        ),
        "split_manifest": str(args.split_csv),
        "crop_mode": args.crop_mode,
        "aspect_balanced_sampling": args.aspect_balance,
        "validation_aspect_groups": val_groups,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(
        "Checkpoint tốt nhất: {} (epoch {}, validation macro F1={:.4f})".format(
            str(checkpoint),
            saved["epoch"],
            saved["val_macro_f1"],
        ),
        flush=True,
    )
    print(
        "Validation accuracy={:.4f}; macro F1={:.4f}; "
        "không tải hay đánh giá ảnh/features test. Kết quả chi tiết: {}".format(
            val_accuracy,
            val_report["macro avg"]["f1-score"],
            str(output_dir),
        ),
        flush=True,
    )

if __name__ == "__main__":
    main()