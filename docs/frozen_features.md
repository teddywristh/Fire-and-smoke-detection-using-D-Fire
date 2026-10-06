# Frozen Transfer Learning Branch Documentation

This document summarizes the transfer learning experiments with a frozen backbone conducted for the Forest Fire C4 project, along with preprocessing/aspect tests and an analysis of evaluation limitations. The test metrics from older runs are kept as historical results for reference; they are **no longer independent test estimates**, as the same test set has been exposed during multiple configuration trials (model selection leakage).

For the dataset layout, manifest creation, and preprocessing recipes specific to this branch, see [data_preprocessing_ff.md](../data_preprocessing_ff.md).

## 1. Problem and Data

The task is to classify images into four classes:

| Label | Meaning |
| --- | --- |
| `fire` | Fire present, no clear smoke visible |
| `nofire` | Neither fire nor smoke visible |
| `smoke` | Smoke present, no fire visible |
| `smokefire` | Both smoke and fire present |

The standard manifest is `data/split.csv`, consisting of 3,200 train, 800 validation, and 800 test images (each class has 800/200/200 images, respectively). The processed images for the general pipeline have a size of 224×224. The Xception aspect pipeline reads raw images based on the manifest paths to handle geometry processing internally.

### Aspect Ratio Skew Detection

An audit of the raw images in the original split revealed that the test set has a different geometric distribution compared to the train/validation sets, especially for the `smokefire` class:

| Class | Train: square / wide | Val: square / wide | Test: square / wide |
| --- | --- | --- | --- |
| `fire` | 800 / 0 | 200 / 0 | 200 / 0 |
| `nofire` | 778 / 12 | 199 / 0 | 200 / 0 |
| `smoke` | 35 / 235 | 4 / 66 | 0 / 103 |
| `smokefire` | 766 / 11 | 194 / 1 | 0 / 99 |

The "square" group uses a width/height ratio of 0.90–1.10; the "wide" group uses 1.60–1.95; the rest are categorized as `other`. The aspect group in the test set is heavily correlated with the class: the square group only consists of `fire`/`nofire`, while the wide/other group mainly consists of `smoke`/`smokefire`. Therefore, metrics separated by aspect group merely describe performance on these existing class combinations, and do not prove that aspect ratio is the sole underlying cause of performance variations.

## 2. How Frozen Transfer was Implemented

### ResNet-50, EfficientNet-B0, MobileNetV3-Large, and Xception baseline

The script `src/frozen_transfer.py` initializes an ImageNet pretrained backbone, sets `requires_grad=False` for the feature extractor, and only trains the classification head. The backbone is run once to cache the train/validation feature vectors; a horizontally flipped view of the training data is also cached to alternate augmentation when training the head.

* Loss: `CrossEntropyLoss`.
* Optimizer: Adam, default learning rate `1e-3`.
* Scheduler: `ReduceLROnPlateau` monitoring validation loss.
* Checkpoint: Selected based on the lowest validation loss.
* After selecting the checkpoint, the old script runs a test evaluation and outputs a classification report/confusion matrix.
* Architectures supported in code: `resnet50`, `efficientnet_b0`, `mobilenet_v3_large`, `xception`. In the current artifacts, full runs are documented for ResNet-50 and Xception.

Commands to run the baselines:

```powershell
python -m src.frozen_transfer --architecture resnet50 --epochs 20 --batch-size 32
python -m src.frozen_transfer --architecture xception --epochs 20 --batch-size 32

```

ResNet/EfficientNet/MobileNet use a standard 224×224 preprocessing with ImageNet mean/std `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`. The Xception baseline uses 299×299, but the baseline script also utilizes the standard ImageNet mean/std.

### Xception aspect robust, letterbox and feature cache

The script `src.xception_aspect_frozen.py` is a separate pipeline that reads raw images, applies letterboxing to make them square, caches frozen features by split, and reports additional metrics by aspect group. The backbone is `timm`'s `legacy_xception`, ImageNet pretrained, using pooling to output a 2048-dimensional feature vector. Normalization is taken from the pretrained Xception config: mean/std `(0.5, 0.5, 0.5)`, scaling RGB to `[-1, 1]`.

The classification head used in the letterbox runs:

```text
2048 → Dropout(0.45) → Linear(512) → ReLU → BatchNorm
      → Dropout(0.2) → Linear(4)

```

Trained with AdamW (`lr=5e-4`, weight decay `1e-4`), ReduceLROnPlateau, and early stopping; the checkpoint is selected by validation macro F1, using validation loss as a tie-breaker/scheduler monitor. Training augmentation includes random resized crop with varying aspect/scale, mild affine transforms, horizontal flip, and brightness/contrast jitter. One view of each training image is cached per run.

Command to run the 299×299 letterbox (older runs):

```powershell
python -m src.xception_aspect_frozen --epochs 30 --batch-size 32 --cpu-threads 4

```

Supported experimental sizes include 64, 224, 256, 299, 320, and 384. Caches/checkpoints/reports are separated by image size.

## 3. Experiment Results

### Historical Results on the Original Test Set

| Experiment | Input / Preprocessing | Best Epoch (Val) | Test Accuracy | Test Macro F1 | `smokefire` Recall / F1 |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 frozen baseline | 224, direct resize; ImageNet norm | 20 | 78.38% | 78.44% | 61.00% / 59.66% |
| Xception frozen baseline | 299, resize; Standard ImageNet norm | 18 | 75.88% | 75.65% | 59.00% / 57.70% |
| Xception aspect robust | 299, letterbox; Xception norm | 14 | 70.50% | 69.30% | 36.50% / 40.11% |
| Xception aspect robust | 256, letterbox; Xception norm | 23 | 71.38% | 70.12% | 33.50% / 38.29% |
| Xception aspect robust | 224, letterbox; Xception norm | 16 | 71.50% | 70.17% | 34.50% / 39.66% |
| Xception aspect robust | 64, letterbox; Xception norm | 9 | 64.88% | 62.60% | 19.50% / 23.78% |

The 64/224/256/299 data points were all calculated on the *same old test set*. This table serves merely as a historical log; it should not be used to claim the optimal scale for new data. 64×64 clearly degrades performance. 224/256 scored slightly higher than 299 overall in older runs, but 299 yielded the highest `smokefire` F1/recall within this letterbox group.

Furthermore, comparing the Xception baseline with the Xception aspect-robust run is not a controlled experiment isolating the letterbox effect: besides preprocessing, they differ in normalization, head architecture/training setup, and feature-caching protocols. The baseline uses standard ImageNet mean/std; the aspect run uses the correct Xception mean/std of 0.5/0.5. Therefore, metric differences between these rows cannot be attributed solely to letterboxing.

ResNet-50 remains the strongest frozen baseline among the 4-class classification runs documented here. The main errors for ResNet on the historical test set are `fire → smokefire` (70 images) and `smokefire → smoke` (77 images). With the Xception letterbox 299, the largest error is `smokefire → smoke` (119/200). `nofire` is generally the easiest class; distinguishing `fire` from `smokefire` and `smoke` from `smokefire` are consistently the most difficult tasks.

### ResNet Frozen Variants

| Variant | Val Macro F1 | Test Accuracy | Test Macro F1 | Notes |
| --- | --- | --- | --- | --- |
| ResNet-50 4-class baseline | 92.77% | 78.38% | 78.44% | Frozen baseline |
| Dual-head fire/smoke, combined to 4 classes | 92.05% | 75.75% | 75.01% | No better than baseline on old test |
| Post-training logit bias calibration | 94.22% | 77.88% | 77.66% | Val increased but old test decreased |

Calibration bias was optimized on the validation set (`fire +0.11`, `nofire +0.05`, `smoke +0.40`, `smokefire -0.20`). The results indicate that increasing the validation score does not guarantee improvements in the test domain. Detailed results are located in `results/transfer_frozen/resnet50_multilabel/` and `results/transfer_frozen/resnet50_calibrated/`.

### New Aspect-Aware Split and Xception Square Crop

To introduce wide `smokefire` images into the validation set, `src.create_aspect_split` reshuffles **only the old train+validation pool**, maintaining exact test membership while checking for raw SHA256 duplicates. The new manifest is `data/split_aspect.csv`; each class contains 800 train and 200 validation images. Given the current data source, wide `smokefire` images are split into 8 train and 4 validation. While this creates a preliminary check for the wide group, the validation set still lacks sufficient wide `smokefire` images to provide a reliable estimate or fully represent the test distribution.

A new experiment uses Xception frozen at 224×224:

* Train: random square crop, mild affine, flip, brightness/contrast.
* Validation: deterministic center square crop.
* Train sampling: weights based on the inverse square root of the frequency of each `(class, aspect group)` pair.
* Checkpoint selected by validation macro F1; best epoch was 28/30.
* **No feature extraction, prediction, or metric calculation was performed on the test set.**

| Validation Metric | Score |
| --- | --- |
| Accuracy | 89.88% |
| Macro Precision | 90.01% |
| Macro Recall | 89.88% |
| Macro F1 | 89.84% |
| `smokefire` Precision / Recall / F1 | 86.56% / 80.50% / 83.42% |
| `smokefire` Wide Recall | 50.00% (2/4 images) |

Do not directly compare these validation scores with the test scores of older runs: the split, preprocessing, and sampler have all changed. Four wide `smokefire` validation images are far too few to infer robustness on wide-angle cameras. It is necessary to acquire additional wide-angle images with labels from a source independent of the Kaggle test set.

## 4. Why Letterbox Hasn't Eliminated the Aspect Shortcut

Letterboxing preserves object geometry but fills empty spaces on the sides or top/bottom with black borders. The size and position of these black regions directly depend on the source aspect ratio. Consequently, even without image stretching, the model can infer whether the original image was square or wide based on the padding pattern. If that aspect ratio correlates with a class or camera source, the padding essentially becomes a shortcut signal.

Square cropping removes padding borders and forces all inputs into the same square geometry. The trade-off is that cropping may cut off objects or context at the edges of the image. Furthermore, it does not eliminate other domain signals such as camera type, lighting, color profiles, resolution, or background context. Cropping should be viewed strictly as a direct mitigation of the aspect signal, not a guarantee of removing domain shift entirely.

## 5. Notes on Test Leakage and Interpreting Results

The original test set has been used to observe the outcomes of multiple configurations (ResNet, Xception, letterboxing, various scales, calibration, and label variants). Therefore, selecting a resolution (e.g., 224/256/299) or preprocessing method based on those test metrics constitutes data leakage at the model selection level. The old metrics remain useful for diagnosing observed errors, but they are no longer an independent final evaluation.

The new square crop pipeline selects the model purely on `data/split_aspect.csv` and does not touch test images or features. However, retaining the Kaggle test set does not erase the influence of previous test set exposures. To achieve a truly independent final evaluation, a new test set from a different source/camera is required, or new test data must be collected before deciding on the final model; once the model and new preprocessing are locked in, that evaluation should be run exactly once.

The diagnostic code block utilizing `frames["test"]` to extract test features and print accuracy/errors is an evaluation. Do not run this block iteratively during the experimentation cycle; the same analysis can be applied to the new validation set to find confident errors, but avoid using it to repeatedly select configurations based on test data.

## 6. Running the New Branch and Artifacts

Create the aspect-aware split (reproducible with the project's default seed):

```powershell
python -m src.create_aspect_split --output data/split_aspect.csv

```

Run Xception frozen with square-crop 224×224 and balanced sampling across class/aspect:

```powershell
python -m src.xception_aspect_frozen --split-csv data/split_aspect.csv --crop-mode square --aspect-balance --image-size 224 --epochs 30 --batch-size 32 --cpu-threads 4

```

Artifacts generated by the new run:

* Manifest: `data/split_aspect.csv`.
* Code: `src/create_aspect_split.py`, `src/xception_aspect_frozen.py`.
* Report: `results/transfer_frozen/xception_square_aspect_balanced_224/summary.md`.
* History/metrics: `training_history.csv`, `validation_classification_report.json`, `validation_confusion_matrix.csv/.png`, `validation_aspect_group_metrics.json` (inside the report directory).
* Checkpoint: `models/transfer_frozen/xception_square_aspect_balanced_224_best.pt`.
* Feature cache: `data/cache/xception_square_aspect_balanced_224_*_v2/`.

Heavy checkpoints and caches are kept local via `.gitignore`; the split manifest and the script to generate it are committed to ensure reproducibility. Do not overwrite older ResNet artifacts.

## 7. Fine-tuning is the Next Branch, Not Frozen

A separate experiment was conducted unfreezing `layer4` of ResNet-50 starting from the frozen checkpoint: Validation macro F1 was 95.26%, old test accuracy 79.88%, macro F1 80.31%, and `smokefire` recall/F1 were 71.00%/64.69%. This falls under the fine-tuned branch, not frozen; the old test set evaluation suffers from the same leakage limitations mentioned above. Refer to `results/transfer_finetuned/resnet50/summary.md`.

For future experiments, the priority is to increase the volume of wide `smokefire` images from independent sources while maintaining a representative validation set. Only then should comparisons be made regarding unfreezing the final blocks or testing different backbones; if the domain or split remains skewed, a larger model might only improve validation scores without ensuring robustness outside the domain.
