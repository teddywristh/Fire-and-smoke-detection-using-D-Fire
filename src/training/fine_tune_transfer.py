"""Fine-tune the final ResNet-50 block from the best frozen checkpoint.

Validation macro F1 selects the checkpoint; test data is evaluated once after
training. Earlier backbone blocks and BatchNorm running statistics stay frozen.

Run: ``python -m src.training.fine_tune_transfer --epochs 8 --batch-size 32``
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
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = Path(row.processed_filepath)
        if not path.is_absolute():
            path = config.PROJECT_ROOT / path
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row.class_index)


def epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
          device: torch.device, optimizer: torch.optim.Optimizer | None = None
          ) -> tuple[float, float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    if training:
        # Small batches should not rewrite the pretrained BN running statistics.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(logits.argmax(1).detach().cpu().tolist())
    y_true = np.asarray(labels_all)
    y_pred = np.asarray(predictions_all)
    return (total_loss / len(loader.dataset), float((y_true == y_pred).mean()),
            y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "models/transfer_frozen/resnet50_best.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest = pd.read_csv(config.SPLIT_CSV_PATH)
    required = {"processed_filepath", "split", "class_index", "status"}
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"split.csv missing columns: {sorted(missing_columns)}")
    manifest = manifest[manifest.status == "ok"].copy()
    manifest["class_index"] = manifest.class_index.astype(int)
    missing_paths = []
    for relative in manifest.processed_filepath:
        path = Path(relative)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.is_file():
            missing_paths.append(path)
    if missing_paths:
        raise FileNotFoundError(
            f"Missing {len(missing_paths)} processed images; run `python -m src.data.data_loader`. "
            f"Example: {missing_paths[0]}"
        )

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    datasets = {
        "train": ManifestDataset(manifest[manifest.split == "train"], train_transform),
        "val": ManifestDataset(manifest[manifest.split == "val"], eval_transform),
        "test": ManifestDataset(manifest[manifest.split == "test"], eval_transform),
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                          num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(config.CLASS_NAMES))
    saved_frozen = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved_frozen["model_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    model = model.to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": args.backbone_lr},
        {"params": model.fc.parameters(), "lr": args.head_lr},
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    output_dir = PROJECT_ROOT / "results/transfer_finetuned/resnet50"
    checkpoint_dir = PROJECT_ROOT / "models/transfer_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "resnet50_best.pt"
    history: list[dict[str, float | int]] = []
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    stale_epochs = 0

    for current_epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = epoch(
            model, loaders["train"], criterion, device, optimizer)
        val_loss, val_accuracy, val_true, val_pred = epoch(
            model, loaders["val"], criterion, device)
        val_macro_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        scheduler.step(val_loss)
        history.append({"epoch": current_epoch, "train_loss": train_loss,
                        "train_accuracy": train_accuracy, "val_loss": val_loss,
                        "val_accuracy": val_accuracy, "val_macro_f1": val_macro_f1,
                        "backbone_lr": optimizer.param_groups[0]["lr"],
                        "head_lr": optimizer.param_groups[1]["lr"]})
        print(f"Epoch {current_epoch:02d}/{args.epochs} train_loss={train_loss:.4f} "
              f"train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f}", flush=True)
        improved = (val_macro_f1 > best_macro_f1 or
                    (val_macro_f1 == best_macro_f1 and val_loss < best_val_loss))
        if improved:
            best_macro_f1, best_val_loss = val_macro_f1, val_loss
            stale_epochs = 0
            torch.save({"architecture": "resnet50", "class_names": config.CLASS_NAMES,
                        "model_state_dict": model.state_dict(), "epoch": current_epoch,
                        "val_loss": val_loss, "val_macro_f1": val_macro_f1,
                        "seed": args.seed}, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {current_epoch} epochs.", flush=True)
                break

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    best = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    _, test_accuracy, test_true, test_pred = epoch(
        model, loaders["test"], criterion, device)
    report = classification_report(test_true, test_pred,
                                   labels=list(range(len(config.CLASS_NAMES))),
                                   target_names=config.CLASS_NAMES, output_dict=True,
                                   zero_division=0)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    matrix = confusion_matrix(test_true, test_pred, labels=list(range(len(config.CLASS_NAMES))))
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *config.CLASS_NAMES])
        for name, row in zip(config.CLASS_NAMES, matrix.tolist()):
            writer.writerow([name, *row])
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(config.CLASS_NAMES)), yticks=range(len(config.CLASS_NAMES)),
             xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
             xlabel="Predicted label", ylabel="True label",
             title="Fine-tuned ResNet-50 - Test")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    print(f"Best checkpoint: {checkpoint} (epoch {best['epoch']}, "
          f"validation macro F1={best['val_macro_f1']:.4f})")
    print(f"Test accuracy: {test_accuracy:.4f}; test macro F1: "
          f"{report['macro avg']['f1-score']:.4f}; reports: {output_dir}")


if __name__ == "__main__":
    main()
