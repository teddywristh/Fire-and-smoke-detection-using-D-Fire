# -*- coding: utf-8 -*-
"""
Tạo tập split train/validation có phân tầng theo class và aspect, không dùng test.
Chạy:
    python -m src.data.create_aspect_split --output data/split_aspect.csv
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
import pandas as pd
import config

def aspect_group(width, height):
    """
    Trả về:
        - "square" nếu tỷ lệ trong [0.9, 1.1]
        - "wide"   nếu tỷ lệ trong [1.6, 1.95]
        - "other"  cho các trường hợp còn lại
    """
    ratio = float(width) / float(height)
    if 0.9 <= ratio <= 1.1:
        return "square"
    if 1.6 <= ratio <= 1.95:
        return "wide"
    return "other"

def class_quotas(frame, fraction, min_wide_smokefire_val):
    """
    frame:
        DataFrame chứa các hàng của một class (label cố định), đã có cột aspect_group.
    fraction:
        Tỷ lệ mẫu muốn đưa vào validation (ví dụ 0.2).
    min_wide_smokefire_val:
        Số lượng tối thiểu ảnh “wide” thuộc class smokefire cần có trong validation.
    trả về:
        Dict {aspect_group: số_lượng_validation_cần_lấy}.
    """
    counts = {}
    for group in frame.aspect_group.unique():
        sub = frame[frame.aspect_group == group]
        counts[group] = len(sub)
    target = round(len(frame) * fraction)
    exact = {}
    for group, count in counts.items():
        exact[group] = count * fraction
    quotas = {}
    for group, value in exact.items():
        quotas[group] = int(value)
    remainder_order = sorted(
        counts.keys(),
        key=lambda g: (exact[g] - quotas[g], counts[g]),
        reverse=True
    )
    while sum(quotas.values()) < target:
        for group in remainder_order:
            if quotas[group] < counts[group]:
                quotas[group] += 1
                break
    while sum(quotas.values()) > target:
        for group in reversed(remainder_order):
            if quotas[group] > 0:
                quotas[group] -= 1
                break
    labels_in_frame = []
    for lab in frame.label.unique():
        labels_in_frame.append(lab)

    if "smokefire" in labels_in_frame and counts.get("wide", 0) > 0:
        desired = min(min_wide_smokefire_val, counts["wide"])
        current_wide = quotas.get("wide", 0)

        if current_wide < desired:
            delta = desired - current_wide
            quotas["wide"] = desired

            other_groups = [g for g in counts.keys() if g != "wide"]
            other_groups_sorted = sorted(other_groups, key=lambda g: counts[g], reverse=True)

            for group in other_groups_sorted:
                if delta <= 0:
                    break
                take = min(delta, quotas[group])
                quotas[group] -= take
                delta -= take
    if sum(quotas.values()) != target:
        raise ValueError(
            "Không thể phân bổ số hàng valid set và giữ nguyên số mẫu từng class"
        )
    return quotas

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=config.SPLIT_CSV_PATH,
        help="File path manifest đầu vào"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PROJECT_ROOT / "data/split_aspect.csv",
        help="File path to split_aspect.csv đầu ra"
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Tỷ lệ mẫu từ development pool đưa vào validation"
    )
    parser.add_argument(
        "--min-wide-smokefire-val",
        type=int,
        default=4,
        help="Số lượng tối thiểu ảnh wide thuộc class smokefire trong validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Seed cho random"
    )
    args = parser.parse_args()
    if not (0 < args.validation_fraction < 1):
        parser.error("--validation-fraction phải nằm trong khoảng (0, 1)")
    if args.min_wide_smokefire_val < 0:
        parser.error("--min-wide-smokefire-val không được âm")

    manifest = pd.read_csv(args.input)

    required_cols = {"split", "label", "width", "height", "status", "raw_sha256"}
    missing_cols = required_cols.difference(manifest.columns)
    if missing_cols:
        raise ValueError(
            "Manifest đầu vào thiếu các cột: {}".format(sorted(missing_cols))
        )

    valid_rows = []
    for idx, row in manifest.iterrows():
        if row.status == "ok":
            valid_rows.append(idx)
    valid = manifest.loc[valid_rows].copy()

    pool_rows = []
    test_rows = []
    for idx, row in valid.iterrows():
        if row.split in ["train", "val"]:
            pool_rows.append(idx)
        elif row.split == "test":
            test_rows.append(idx)

    pool = valid.loc[pool_rows].copy()
    test_holdout = valid.loc[test_rows].copy()

    if len(pool) == 0 or len(test_holdout) == 0:
        raise ValueError(
            "Cần có cả development pool (train+val) và test holdout khác rỗng."
        )

    aspect_groups_list = []
    for _, row in pool.iterrows():
        ag = aspect_group(row.width, row.height)
        aspect_groups_list.append(ag)
    pool = pool.copy()
    pool["aspect_group"] = aspect_groups_list
    rng = random.Random(args.seed)

    validation_indices = []
    group_number = 0

    labels_sorted = sorted(list(pool.label.unique()))
    for label in labels_sorted:
        class_mask = [i for i, lab in enumerate(pool.label) if lab == label]
        class_frame = pool.iloc[class_mask].copy()

        quotas = class_quotas(
            class_frame,
            args.validation_fraction,
            args.min_wide_smokefire_val
        )

        for group in sorted(quotas.keys()):
            group_mask = [
                i for i, ag in enumerate(class_frame.aspect_group) if ag == group
            ]
            group_frame = class_frame.iloc[group_mask]
            rows_indices = group_frame.index.tolist()
            rng.seed(args.seed + group_number)
            rng.shuffle(rows_indices)
            num_val = quotas[group]
            validation_indices.extend(rows_indices[:num_val])
            group_number += 1

    validation_set = set(validation_indices)

    train_rows = []
    val_rows = []
    for idx in pool.index:
        if idx in validation_set:
            val_rows.append(idx)
        else:
            train_rows.append(idx)

    train_pool = pool.loc[train_rows].copy()
    val_pool = pool.loc[val_rows].copy()

    if "aspect_group" in train_pool.columns:
        train_pool = train_pool.drop(columns=["aspect_group"])
    if "aspect_group" in val_pool.columns:
        val_pool = val_pool.drop(columns=["aspect_group"])

    train_pool = train_pool.copy()
    val_pool = val_pool.copy()
    test_holdout = test_holdout.copy()

    train_pool["source_split"] = train_pool["split"]
    val_pool["source_split"] = val_pool["split"]
    test_holdout["source_split"] = "test"

    train_pool["split"] = "train"
    val_pool["split"] = "val"
    test_holdout["split"] = "test"

    combined_list = [train_pool, val_pool, test_holdout]
    combined = pd.concat(combined_list, ignore_index=True)

    hash_sets = {}
    for split_name in ["train", "val", "test"]:
        sub = combined[combined.split == split_name]
        hashes = set(sub.raw_sha256.dropna().tolist())
        hash_sets[split_name] = hashes

    if (hash_sets["train"] & hash_sets["val"]) or \
       (hash_sets["train"] & hash_sets["test"]) or \
       (hash_sets["val"] & hash_sets["test"]):
        raise ValueError(
            "Phát hiện trùng raw_sha256 giữa các split được sinh ra."
        )

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    combined.to_csv(args.output, index=False)
    audit_mask = [
        i for i, sp in enumerate(combined.split) if sp in ["train", "val"]
    ]
    audit = combined.iloc[audit_mask].copy()

    audit_aspect = []
    for _, row in audit.iterrows():
        audit_aspect.append(aspect_group(row.width, row.height))
    audit["aspect_group"] = audit_aspect

    print("Đã ghi development split vào: {}".format(str(args.output)))
    print("Bảng phân bố theo split, label và aspect_group:")
    print(pd.crosstab([audit.split, audit.label], audit.aspect_group))
    print("Số lượng mẫu theo split và label:")
    print(pd.crosstab(audit.split, audit.label))


if __name__ == "__main__":
    main()