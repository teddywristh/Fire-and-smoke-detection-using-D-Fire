# -*- coding: utf-8 -*-
"""
Huấn luyện transfer learning với backbone ImageNet bị đóng băng (frozen)
trên bộ dữ liệu Forest Fire C4.
Sử dụng trực tiếp file manifest split chung của dự án.
Dữ liệu test chỉ được đánh giá sau khi đã chọn checkpoint tốt nhất
theo validation loss.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config

class ManifestDataset(Dataset):
    """
    Dataset ảnh được xây dựng từ file manifest split.csv của dự án.
    Mỗi hàng trong manifest tương ứng với một ảnh đã qua tiền xử lý.
    """
    def __init__(self, frame, transform):
        """
        frame: DataFrame chứa các hàng của một split (train/val/test).
        transform: phép biến đổi ảnh (resize, normalize, v.v.).
        """
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = Path(row.processed_filepath)

        if not image_path.is_absolute():
            image_path = config.PROJECT_ROOT / image_path

        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = int(row.class_index)
        return tensor, label

def build_model(architecture, num_classes, pretrained=True):
    """
    Tạo mô hình với backbone pretrained (ImageNet) và head phân lớp mới.
    architecture: tên mô hình ("resnet50", "efficientnet_b0", "mobilenet_v3_large", "xception").
    num_classes: số lớp đầu ra (số class trong bài toán).
    pretrained: có nạp trọng số pretrained hay không.
    mô hình nn.Module đã được cấu hình.
    """
    if architecture == "xception":
        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "Mô hình Xception yêu cầu gói `timm`. "
                "Hãy cài đặt các yêu cầu của dự án trước."
            ) from error

        model = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=num_classes,
        )

        for parameter in model.parameters():
            parameter.requires_grad = False
        head = model.get_classifier()
        for parameter in head.parameters():
            parameter.requires_grad = True
        return model

    constructors = {
        "resnet50": (
            models.resnet50,
            models.ResNet50_Weights.DEFAULT,
            "fc",
        ),
        "efficientnet_b0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.DEFAULT,
            "classifier",
        ),
        "mobilenet_v3_large": (
            models.mobilenet_v3_large,
            models.MobileNet_V3_Large_Weights.DEFAULT,
            "classifier",
        ),
    }
    constructor, weights, head_name = constructors[architecture]
    model = constructor(weights=weights if pretrained else None)

    for parameter in model.parameters():
        parameter.requires_grad = False

    if head_name == "fc":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    else:
        old_head = model.classifier[-1]
        model.classifier[-1] = nn.Linear(old_head.in_features, num_classes)
        head = model.classifier[-1]
    for parameter in head.parameters():
        parameter.requires_grad = True
    return model

def classification_head(model, architecture):
    """
    Lấy module head phân lớp của mô hình.
    """
    if architecture == "resnet50":
        return model.fc
    if architecture == "xception":
        return model.get_classifier()
    return model.classifier


def backbone_features(model, architecture, images):
    """
    model: mô hình đã được build.
    architecture: tên mô hình.
    images: tensor ảnh đầu vào.
    tensor đặc trưng (batch_size, feature_dim).
    """
    if architecture == "resnet50":
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
    elif architecture == "xception":
        features = model.forward_features(images)
        return model.forward_head(features, pre_logits=True)
    else:
        # EfficientNet, MobileNet: dùng model.features + avgpool
        x = model.features(images)
        x = model.avgpool(x)

    return torch.flatten(x, 1)

@torch.no_grad()
def cache_features(
    model,
    architecture,
    loader,
    device,
    include_flipped=False,
):
    """
    model: mô hình đã được build.
    architecture: tên mô hình.
    loader: DataLoader chứa ảnh và nhãn.
    device: thiết bị tính toán.
    include_flipped: có cache thêm flipped features hay không.
    trả về:
        - original_features: tensor đặc trưng gốc.
        - flipped_features: tensor đặc trưng lật ngang (hoặc None).
        - labels: tensor nhãn.
    """
    model.eval()

    original_parts = []
    flipped_parts = []
    label_parts = []

    for images, labels in loader:
        images = images.to(device)
        orig_feats = backbone_features(model, architecture, images).cpu()
        original_parts.append(orig_feats)
        if include_flipped:
            flipped = torch.flip(images, dims=[3])
            flip_feats = backbone_features(model, architecture, flipped).cpu()
            flipped_parts.append(flip_feats)
        label_parts.append(labels.cpu())
    original_features = torch.cat(original_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    if include_flipped:
        flipped_features = torch.cat(flipped_parts, dim=0)
    else:
        flipped_features = None
    return original_features, flipped_features, labels

def run_head_epoch(
    model,
    architecture,
    features,
    flipped_features,
    labels,
    criterion,
    device,
    optimizer,
    batch_size,
):
    """
    model: mô hình đã được build.
    architecture: tên mô hình.
    features: tensor đặc trưng gốc (num_samples, feature_dim).
    flipped_features: tensor đặc trưng lật ngang (hoặc None).
    labels: tensor nhãn.
    criterion: hàm loss.
    device: thiết bị tính toán.
    optimizer: optimizer (hoặc None nếu chỉ đánh giá).
    batch_size: kích thước batch.
    trả về:
        - average_loss: loss trung bình trên toàn bộ mẫu.
        - accuracy: độ chính xác trên toàn bộ mẫu.
    """
    training = optimizer is not None
    head = classification_head(model, architecture)
    head.train(training)
    if training:
        order = torch.randperm(len(labels))
    else:
        order = torch.arange(len(labels))

    total_loss = 0.0
    correct = 0

    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch_features = features[indices]

        if training and flipped_features is not None:
            choose_flip = torch.rand(len(indices)) < 0.5
            batch_features = batch_features.clone()
            batch_features[choose_flip] = flipped_features[indices[choose_flip]]

        batch_features = batch_features.to(device)
        batch_labels = labels[indices].to(device)

        if training:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()
        with context:
            logits = head(batch_features)
            loss = criterion(logits, batch_labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(indices)
        correct += (logits.argmax(1) == batch_labels).sum().item()
    average_loss = total_loss / len(labels)
    accuracy = correct / len(labels)
    return average_loss, accuracy

def main():
    """
      - Đọc manifest, kiểm tra dữ liệu.
      - Xây dựng DataLoader cho train/val/test.
      - Build mô hình, đóng băng backbone.
      - Cache đặc trưng backbone.
      - Huấn luyện head phân lớp.
      - Lưu checkpoint tốt nhất theo validation loss.
      - Đánh giá trên test và ghi báo cáo, ma trận nhầm lẫn.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=["resnet50", "efficientnet_b0", "mobilenet_v3_large", "xception"],
        default="resnet50",
        help="Kiến trúc mô hình cần huấn luyện",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Số epoch huấn luyện",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Kích thước batch",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Tốc độ học",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Số worker cho DataLoader",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Seed cho random",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Không nạp trọng số ImageNet (hữu ích khi offline).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Tiếp tục huấn luyện head từ một checkpoint đã lưu.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    manifest = pd.read_csv(config.SPLIT_CSV_PATH)

    required_cols = {
        "processed_filepath",
        "split",
        "label",
        "class_index",
        "status",
    }
    missing_cols = required_cols.difference(manifest.columns)
    if missing_cols:
        raise ValueError(
            "File split.csv thiếu các cột bắt buộc: {}".format(
                sorted(missing_cols)
            )
        )

    ok_mask = manifest.status == "ok"
    manifest = manifest[ok_mask].copy()
    manifest["class_index"] = manifest.class_index.astype(int)

    missing_images = []
    for relative_path in manifest.processed_filepath:
        image_path = Path(relative_path)
        if not image_path.is_absolute():
            image_path = config.PROJECT_ROOT / image_path
        if not image_path.is_file():
            missing_images.append(image_path)

    if missing_images:
        examples = "\n".join(
            "  - {}".format(str(path))
            for path in missing_images[:5]
        )
        raise FileNotFoundError(
            "Manifest tham chiếu tới {} ảnh đã xử lý bị thiếu. "
            "Ví dụ:\n{}\n"
            "Hãy chuẩn bị dataset đã xử lý trước bằng lệnh: "
            "`python -m src.data.data_loader`. "
            "Thư mục dataset thô hiện tại là: {}".format(
                len(missing_images),
                examples,
                str(config.DATASET_ROOT),
            )
        )

    for split in ("train", "val", "test"):
        sub = manifest[manifest.split == split]
        if sub.empty:
            raise ValueError(
                "Không tìm thấy hàng hợp lệ nào cho split '{}' trong file {}".format(
                    split,
                    str(config.SPLIT_CSV_PATH),
                )
            )

    input_size = config.IMAGE_SIZE
    if args.architecture == "xception":
        # Xception pretrained dùng 299x299
        input_size = (299, 299)

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    eval_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize,
    ])

    datasets = {}
    for split in ("train", "val", "test"):
        sub_manifest = manifest[manifest.split == split]
        datasets[split] = ManifestDataset(sub_manifest, eval_transform)
    loaders = {}
    for name, ds in datasets.items():
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        loaders[name] = loader
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_model(
        args.architecture,
        len(config.CLASS_NAMES),
        pretrained=(not args.no_pretrained),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    output_dir = (
        config.PROJECT_ROOT
        / "results"
        / "transfer_frozen"
        / args.architecture
    )
    checkpoint_dir = (
        config.PROJECT_ROOT
        / "models"
        / "transfer_frozen"
    )

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "{}_best.pt".format(args.architecture)
    train_features, flipped_train_features, train_labels = cache_features(
        model,
        args.architecture,
        loaders["train"],
        device,
        include_flipped=True,
    )
    val_features, _, val_labels = cache_features(
        model,
        args.architecture,
        loaders["val"],
        device,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if args.resume:
        saved = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(saved["model_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_val_loss = float(saved["val_loss"])
        history.append(
            {
                "epoch": int(saved["epoch"]),
                "train_loss": None,
                "train_accuracy": None,
                "val_loss": best_val_loss,
                "val_accuracy": None,
                "learning_rate": args.learning_rate,
            }
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_accuracy = run_head_epoch(
            model,
            args.architecture,
            train_features,
            flipped_train_features,
            train_labels,
            criterion,
            device,
            optimizer,
            args.batch_size,
        )

        val_loss, val_accuracy = run_head_epoch(
            model,
            args.architecture,
            val_features,
            None,
            val_labels,
            criterion,
            device,
            None,
            args.batch_size,
        )

        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            "Epoch {:02d}/{:d} "
            "train_loss={:.4f} train_acc={:.4f} "
            "val_loss={:.4f} val_acc={:.4f}".format(
                epoch,
                args.epochs,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "architecture": args.architecture,
                    "class_names": config.CLASS_NAMES,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    test_features, _, test_labels = cache_features(
        model,
        args.architecture,
        loaders["test"],
        device,
    )
    y_true = []
    y_pred = []

    with torch.no_grad():
        head = classification_head(model, args.architecture)
        for start in range(0, len(test_labels), args.batch_size):
            batch_features = test_features[start:start + args.batch_size]
            logits = head(batch_features.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(
                test_labels[start:start + args.batch_size].tolist()
            )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(config.CLASS_NAMES))),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_path = output_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(config.CLASS_NAMES))),
    )

    cm_csv_path = output_dir / "confusion_matrix.csv"
    with cm_csv_path.open("w", newline="", encoding="utf-8") as file:
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
        title="Frozen Transfer Learning — Test",
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

    cm_png_path = output_dir / "confusion_matrix.png"
    figure.savefig(cm_png_path, dpi=160)
    plt.close(figure)

    print(
        "Checkpoint tốt nhất: {} (epoch {}, val_loss={:.4f})".format(
            str(checkpoint_path),
            saved["epoch"],
            saved["val_loss"],
        )
    )
    print(
        "Độ chính xác trên test: {:.4f}; "
        "Kết quả chi tiết trong thư mục: {}".format(
            report["accuracy"],
            str(output_dir),
        )
    )


if __name__ == "__main__":
    main()