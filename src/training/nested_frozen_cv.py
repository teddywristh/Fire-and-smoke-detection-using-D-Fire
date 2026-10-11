"""Nested cross-validation for the frozen ResNet-50/Xception comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from src.training.xception_aspect_frozen import aspect_group, augment_square_image, square_crop


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "nested-frozen-cv-v1"
ARCHITECTURES = {
    "resnet50": "resnet50",
    "xception": "legacy_xception",
}


class PairedCropDataset(Dataset):
    """Make the same train-augmented and deterministic eval views for every image."""

    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.frame)

    @staticmethod
    def to_tensor(image):
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()
        return torch.from_numpy(array / 127.5 - 1.0)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row.filepath)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        train_view = square_crop(image, training=True)
        train_view = train_view.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        train_view = augment_square_image(train_view)

        eval_view = square_crop(image, training=False)
        eval_view = eval_view.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )

        ratio = float(row.width) / float(row.height)
        return (
            self.to_tensor(train_view),
            self.to_tensor(eval_view),
            int(row.class_index),
            ratio,
        )


def feature_cache_path(frame: pd.DataFrame, architecture: str, image_size: int, seed: int):
    columns = ["filepath", "raw_sha256", "class_index", "width", "height"]
    fingerprint = hashlib.sha256(
        (
            PROTOCOL_VERSION
            + architecture
            + str(image_size)
            + str(seed)
            + frame[columns].to_json(orient="records")
        ).encode("utf-8")
    ).hexdigest()[:16]
    return PROJECT_ROOT / "data/cache" / PROTOCOL_VERSION / f"{architecture}_{fingerprint}.npz"


def extract_features(frame, architecture, image_size, batch_size, device, seed):
    cache_path = feature_cache_path(frame, architecture, image_size, seed)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            print(f"Using cached {architecture} features: {cache_path}", flush=True)
            return {name: cache[name] for name in ("train", "eval", "labels", "ratios")}

    random.seed(seed)
    torch.manual_seed(seed)
    dataset = PairedCropDataset(frame, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = timm.create_model(
        ARCHITECTURES[architecture], pretrained=True, num_classes=0
    ).to(device)
    model.eval()

    train_parts = []
    eval_parts = []
    label_parts = []
    ratio_parts = []
    with torch.inference_mode():
        for batch_index, (train_images, eval_images, labels, ratios) in enumerate(loader, 1):
            images = torch.cat((train_images, eval_images), dim=0).to(device)
            features = model(images).float().cpu().numpy()
            batch_len = len(labels)
            train_parts.append(features[:batch_len])
            eval_parts.append(features[batch_len:])
            label_parts.append(labels.numpy())
            ratio_parts.append(ratios.numpy())
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"{architecture} feature batches: {batch_index}/{len(loader)}",
                    flush=True,
                )

    arrays = {
        "train": np.concatenate(train_parts).astype(np.float32, copy=False),
        "eval": np.concatenate(eval_parts).astype(np.float32, copy=False),
        "labels": np.concatenate(label_parts).astype(np.int64, copy=False),
        "ratios": np.concatenate(ratio_parts).astype(np.float32, copy=False),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    del model
    return arrays


def make_head(feature_dim: int, num_classes: int):
    return nn.Sequential(
        nn.Dropout(0.45),
        nn.Linear(feature_dim, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )


def balanced_weights(labels: np.ndarray, ratios: np.ndarray):
    groups = [aspect_group(float(ratio)) for ratio in ratios]
    keys = [(int(label), group) for label, group in zip(labels, groups)]
    counts = pd.Series(keys).value_counts().to_dict()
    weights = np.asarray(
        [1.0 / np.sqrt(counts[key]) for key in keys], dtype=np.float64
    )
    weights /= weights.sum()
    return torch.as_tensor(weights, dtype=torch.double)


def evaluate(head, features, labels, criterion, device, batch_size):
    head.eval()
    logits_parts = []
    loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            x = features[start : start + batch_size].to(device)
            y = labels[start : start + batch_size].to(device)
            logits = head(x)
            loss_sum += criterion(logits, y).item() * len(y)
            logits_parts.append(logits.cpu())
    logits = torch.cat(logits_parts)
    return loss_sum / len(labels), logits


def fit_inner_model(
    train_features,
    train_labels,
    train_ratios,
    val_features,
    val_labels,
    seed,
    args,
    device,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    head = make_head(train_features.shape[1], len(config.CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss()
    weights = balanced_weights(train_labels.numpy(), train_ratios)
    best_f1 = -1.0
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.max_epochs + 1):
        head.train()
        order = torch.multinomial(weights, len(train_labels), replacement=True)
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            if len(indices) < 2:
                continue
            x = train_features[indices].to(device)
            y = train_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(x), y)
            loss.backward()
            optimizer.step()

        val_loss, val_logits = evaluate(
            head, val_features, val_labels, criterion, device, args.batch_size
        )
        val_f1 = f1_score(
            val_labels.numpy(), val_logits.argmax(1).numpy(),
            labels=list(range(len(config.CLASS_NAMES))),
            average="macro", zero_division=0,
        )
        scheduler.step(val_loss)
        if val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_loss):
            best_f1 = float(val_f1)
            best_loss = float(val_loss)
            best_state = copy.deepcopy(head.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    head.load_state_dict(best_state)
    return head


def metrics_for(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(config.CLASS_NAMES))),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        **{
            f"f1_{name}": float(report[name]["f1-score"])
            for name in config.CLASS_NAMES
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=PROJECT_ROOT / "data/split_aspect.csv")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/frozen_nested_cv"
    )
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("outer-folds and inner-folds must both be at least 2")

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.split_csv)
    required = {
        "filepath", "raw_sha256", "split", "label", "class_index",
        "width", "height", "status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame = frame.loc[
        frame.status.eq("ok") & frame.split.isin(["train", "val"])
    ].copy().reset_index(drop=True)
    if frame.raw_sha256.duplicated().any():
        raise ValueError("Development pool contains duplicate raw SHA-256 values")
    y = frame.class_index.astype(int).to_numpy()
    if any(np.count_nonzero(y == index) < args.outer_folds for index in range(len(config.CLASS_NAMES))):
        raise ValueError("Every class must contain at least outer-folds samples")

    print(
        f"Development-only nested CV: n={len(frame)}, outer={args.outer_folds}, "
        f"inner={args.inner_folds}, device={device}; test split excluded.",
        flush=True,
    )
    feature_sets = {}
    for architecture in ARCHITECTURES:
        feature_sets[architecture] = extract_features(
            frame, architecture, args.image_size, args.batch_size, device, args.seed
        )

    outer_cv = StratifiedKFold(
        n_splits=args.outer_folds, shuffle=True, random_state=args.seed
    )
    outer_predictions = {
        architecture: np.zeros((len(frame), len(config.CLASS_NAMES)), dtype=np.float64)
        for architecture in ARCHITECTURES
    }
    outer_fold_ids = np.zeros(len(frame), dtype=np.int64)
    fold_rows = []
    started = time.time()

    for outer_fold, (outer_train, outer_test) in enumerate(
        outer_cv.split(np.zeros(len(y)), y), start=1
    ):
        outer_fold_ids[outer_test] = outer_fold
        print(f"Outer fold {outer_fold}/{args.outer_folds}", flush=True)
        inner_cv = StratifiedKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=args.seed + outer_fold,
        )
        inner_splits = list(
            inner_cv.split(np.zeros(len(outer_train)), y[outer_train])
        )

        for architecture, arrays in feature_sets.items():
            fold_probabilities = []
            for inner_fold, (inner_train_local, inner_val_local) in enumerate(
                inner_splits, start=1
            ):
                inner_train = outer_train[inner_train_local]
                inner_val = outer_train[inner_val_local]
                model_seed = args.seed + outer_fold * 100 + inner_fold
                head = fit_inner_model(
                    torch.from_numpy(arrays["train"][inner_train]),
                    torch.from_numpy(arrays["labels"][inner_train]),
                    arrays["ratios"][inner_train],
                    torch.from_numpy(arrays["eval"][inner_val]),
                    torch.from_numpy(arrays["labels"][inner_val]),
                    model_seed,
                    args,
                    device,
                )
                _, logits = evaluate(
                    head,
                    torch.from_numpy(arrays["eval"][outer_test]),
                    torch.from_numpy(arrays["labels"][outer_test]),
                    nn.CrossEntropyLoss(),
                    device,
                    args.batch_size,
                )
                fold_probabilities.append(torch.softmax(logits, dim=1).numpy())
                del head

            probabilities = np.mean(fold_probabilities, axis=0)
            outer_predictions[architecture][outer_test] = probabilities
            fold_pred = probabilities.argmax(axis=1)
            fold_scores = metrics_for(y[outer_test], fold_pred)
            fold_rows.append(
                {"outer_fold": outer_fold, "architecture": architecture, **fold_scores}
            )
            print(
                f"  {architecture}: outer accuracy={fold_scores['accuracy']:.4f}, "
                f"macro_f1={fold_scores['macro_f1']:.4f}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(args.output_dir / "outer_fold_metrics.csv", index=False)

    prediction_rows = []
    summary_rows = []
    confusion_rows = []
    for architecture, probabilities in outer_predictions.items():
        predicted = probabilities.argmax(axis=1)
        scores = metrics_for(y, predicted)
        for metric, value in scores.items():
            fold_values = fold_metrics.loc[
                fold_metrics.architecture.eq(architecture), metric
            ].to_numpy()
            summary_rows.append(
                {
                    "architecture": architecture,
                    "metric": metric,
                    "oof_score": value,
                    "outer_fold_mean": float(np.mean(fold_values)),
                    "outer_fold_sd": float(np.std(fold_values, ddof=1)),
                    "outer_fold_min": float(np.min(fold_values)),
                    "outer_fold_max": float(np.max(fold_values)),
                    "n_outer_folds": args.outer_folds,
                }
            )
        for index, row in frame.iterrows():
            entry = {
                "sample_id": row.raw_sha256,
                "source_split": row.split,
                "outer_fold": int(outer_fold_ids[index]),
                "true_label": config.CLASS_NAMES[int(y[index])],
                "predicted_label": config.CLASS_NAMES[int(predicted[index])],
            }
            for class_index, name in enumerate(config.CLASS_NAMES):
                entry[f"prob_{name}"] = float(probabilities[index, class_index])
            prediction_rows.append({"architecture": architecture, **entry})
        matrix = confusion_matrix(y, predicted, labels=list(range(len(config.CLASS_NAMES))))
        for true_index, true_name in enumerate(config.CLASS_NAMES):
            for pred_index, pred_name in enumerate(config.CLASS_NAMES):
                confusion_rows.append(
                    {
                        "architecture": architecture,
                        "true_label": true_name,
                        "predicted_label": pred_name,
                        "count": int(matrix[true_index, pred_index]),
                    }
                )

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "oof_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / "oof_predictions.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        args.output_dir / "confusion_matrices.csv", index=False
    )
    config_output = {
        "protocol": PROTOCOL_VERSION,
        "split_manifest": str(args.split_csv),
        "development_rows": len(frame),
        "test_rows_loaded": False,
        "test_metrics_generated": False,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "architectures": list(ARCHITECTURES),
        "inner_fold_ensemble": True,
        "prediction_rule": "mean class probability across inner-fold selected checkpoints",
        "results_runtime_seconds": round(time.time() - started, 2),
    }
    with (args.output_dir / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump(config_output, file, indent=2)
    print(f"Saved nested CV outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
