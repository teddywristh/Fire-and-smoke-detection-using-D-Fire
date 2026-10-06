# Comparative Deep Learning Models for Forest Fire and Smoke Image Classification

## Project Overview

This project develops and compares Deep Learning models for forest fire and smoke image classification.

For the frozen-features protocols, complete experiment history, metrics, and test-set limitations, see [frozen_features.md](frozen_features.md).

The system classifies images into four categories:

- **Fire:** Fire is visible, with no clear smoke.
- **No Fire:** Neither fire nor smoke is visible.
- **Smoke:** Smoke is visible, but no fire can be seen.
- **SmokeFire:** Both smoke and fire are visible.

The project investigates three main Deep Learning approaches:

1. **Basic Neural Network**
2. **Custom Complex CNN**
3. **Transfer Learning**
   - Frozen pretrained backbone
   - Fine-tuned pretrained backbone

Although there are three main approaches, four model configurations are evaluated:

| Model | Description |
|---|---|
| Basic Neural Network | Fully connected baseline model |
| Custom Complex CNN | CNN designed and trained from scratch |
| Transfer Learning – Frozen | Pretrained backbone with only the classification head trained |
| Transfer Learning – Fine-tuned | Pretrained model with selected backbone layers fine-tuned |

All models use the same dataset split and evaluation criteria to ensure a fair comparison.

---

## Objectives

The main objectives of this project are:

- Build a baseline model using a Basic Neural Network.
- Develop a custom CNN for image classification.
- Apply Transfer Learning using a pretrained CNN.
- Evaluate the effect of fine-tuning.
- Compare the performance of different model architectures.
- Analyze common classification errors.
- Use Grad-CAM to understand which image regions influence the final model's predictions.

Special attention is given to confusion between:

- `Fire` and `SmokeFire`
- `Smoke` and `SmokeFire`
- `Fire` predicted as `No Fire`
- `SmokeFire` predicted as `No Fire`

---

## Dataset

The project uses the **Forest Fire Image Classification Dataset** from Kaggle.

Dataset link:

https://www.kaggle.com/datasets/obulisainaren/forest-fire-c4

The dataset contains four classes:

| Class | Description |
|---|---|
| Fire | Images containing visible fire |
| No Fire | Images without fire or smoke |
| Smoke | Images containing smoke without visible fire |
| SmokeFire | Images containing both smoke and fire |

The dataset already provides training, validation, and test sets.

The original split is preserved for the standard model runs. A separate aspect-aware development split is also available for the frozen Xception square-crop experiment; it repartitions only the original train/validation pool and leaves test membership unchanged.

A shared `split.csv` is used to keep the data split consistent across experiments.

---

## Models

### 1. Basic Neural Network

The Basic Neural Network is used as a baseline model.

Example architecture:

```text
Input
→ Flatten
→ Dense
→ Dropout
→ Dense
→ Softmax
```

This model provides a simple reference point for evaluating more advanced CNN-based approaches.

---

### 2. Custom Complex CNN

The Custom CNN is designed and trained from scratch.

Main components include:

- Conv2D
- Batch Normalization
- Max Pooling
- Dropout
- Global Average Pooling
- Dense layers

Example architecture:

```text
Input
→ Conv2D
→ BatchNormalization
→ MaxPooling
→ Dropout
→ Conv2D
→ BatchNormalization
→ MaxPooling
→ Dropout
→ GlobalAveragePooling
→ Dense
→ Softmax
```

The CNN learns spatial features such as flame patterns, smoke textures, colors, edges, and shapes.

---

### 3. Transfer Learning – Frozen Backbone

A pretrained CNN feature extractor is used with a newly initialized classification head. The implemented PyTorch options are **ResNet-50**, **EfficientNet-B0**, **MobileNetV3-Large**, and **Xception**.

```text
Input
→ Pretrained Backbone
→ GlobalAveragePooling
→ Classification Head
→ Dropout
→ Softmax
```

During this stage:

- The pretrained backbone is frozen.
- Only the classification head is trained.

The same pretrained backbone is used for both the Frozen and Fine-tuning experiments.

---

### 4. Transfer Learning – Fine-tuning

Fine-tuning continues from the best Frozen Transfer Learning model.

Selected final layers of the pretrained backbone are unfrozen and trained using a smaller learning rate.

```text
Best Frozen Model
→ Unfreeze Selected Layers
→ Fine-tuning
→ Final Model
```

This experiment evaluates whether adapting pretrained features to the forest fire dataset improves classification performance.

---

## Training Protocol

To ensure a fair comparison:

- Standard runs use the shared `data/split.csv`; the aspect-aware Xception experiment uses `data/split_aspect.csv`.
- Data augmentation is applied only to training data.
- Validation data is used for checkpoint selection; early stopping is enabled in the aspect-aware Xception script.
- Validation is used for checkpoint selection. Several historical frozen-transfer experiments were compared using the original test set, so those test scores are exploratory and are not independent estimates anymore.
- The current square-crop Xception run does not extract features from or score test images.
- A fixed random seed is used where possible.
- The same evaluation metrics are used for all models.

Model-specific preprocessing may be applied when required by pretrained architectures.

---

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

Per-class performance is also analyzed for all four classes.

Special attention is given to critical false-negative cases:

```text
Fire → No Fire
SmokeFire → No Fire
```

---

## Error Analysis

Misclassified images are analyzed to identify common failure patterns.

Important confusion cases include:

```text
Fire ↔ SmokeFire
Smoke ↔ SmokeFire
Fire → No Fire
SmokeFire → No Fire
```

For selected incorrect predictions, the project examines:

- Original image
- True label
- Predicted label
- Prediction confidence

---

## Grad-CAM

Grad-CAM is applied to the final fine-tuned Transfer Learning model.

It is used to visualize which parts of an image contribute most strongly to the model's prediction.

This helps determine whether the model focuses on meaningful regions such as fire and smoke or on irrelevant background features.

---

## Project Structure

```text
forest_fire_classification/
│
├── data/
│   ├── raw/
│   └── split.csv
│
├── notebooks/
│   ├── 01_basic_nn.ipynb
│   ├── 02_custom_cnn.ipynb
│   ├── 03_transfer_frozen.ipynb
│   └── 04_transfer_finetuning.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── augmentation.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── gradcam.py
│
├── models/
│   ├── basic_nn_best.keras
│   ├── custom_cnn_best.keras
│   ├── transfer_frozen_best.keras
│   └── transfer_finetuned_best.keras
│
├── results/
│   ├── basic_nn/
│   ├── custom_cnn/
│   ├── transfer_frozen/
│   ├── transfer_finetuned/
│   └── final_comparison.csv
│
├── demo/
│   └── predict_image.ipynb
│
├── config.py
├── requirements.txt
├── .gitignore
├── README.md
└── main.ipynb
```

---

## Requirements

Main libraries used in the project:

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Jupyter

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd forest_fire_classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download and configure the dataset

Download the Kaggle dataset and keep the folder that directly contains:

```text
train/
val/
test/
```

By default, this project expects the dataset at:

```text
../dataset/Forect Fire/Forest Fire_Dataset
```

If your dataset is stored somewhere else, do not edit `config.py` directly. Copy the local config template:

```bash
copy local_config.example.py local_config.py
```

Then edit `local_config.py` for your own machine:

```python
from pathlib import Path

DATASET_ROOT = Path(r"D:/your/path/to/dataset/Forect Fire/Forest Fire_Dataset")
TESTER_ROOT = Path(r"D:/your/path/to/dataset/Forect Fire/Forest Fire_Tester")
```

`local_config.py` is ignored by Git, so each team member can use a different dataset location without changing tracked project files.

### 4. Prepare the processed dataset

To download the public C4 dataset from Kaggle and prepare it in one step, install
the requirements and run:

```bash
python -m src.download_dataset
```

KaggleHub caches the download under `data/raw/kagglehub/`; the script locates the
dataset's `train/val/test` folders and runs the existing preprocessing pipeline.
Alternatively, if the raw dataset is already on disk, configure `DATASET_ROOT`
as described above and run `python -m src.data_loader`.

The data preparation command (also run automatically by the Kaggle download script) is:

```bash
python -m src.data_loader
```

This creates:

```text
data/split.csv
data/processed/train/
data/processed/val/
data/processed/test/
data/processed/processing_report.md
data/processed/processing_summary.json
```

The pipeline preserves the original train/validation/test split, resizes images to `224x224`, excludes `Forest Fire_Tester` from training/evaluation, and checks for duplicate files by SHA256.

For raw dataset setup and frozen-features preprocessing details, see [data_preprocessing_ff.md](../data_preprocessing_ff.md). The model protocols and complete frozen-run metrics are in [frozen_features.md](frozen_features.md).

### 5. Run the experiments

Run the notebooks:

```text
01_basic_nn.ipynb
02_custom_cnn.ipynb
03_transfer_frozen.ipynb
04_transfer_finetuning.ipynb
```

The Fine-tuning experiment should start from the best Frozen Transfer Learning model.

The PyTorch frozen transfer baseline can be run directly from the shared CSV
manifest (the processed image folders must already exist):

```bash
python -m src.frozen_transfer --architecture resnet50 --epochs 20 --batch-size 32
```

To run the frozen Xception experiment with its ImageNet 299×299 input:

```bash
python -m src.frozen_transfer --architecture xception --epochs 20 --batch-size 32
```

Supported architectures are `resnet50`, `efficientnet_b0`,
`mobilenet_v3_large`, and `xception`. Xception uses `timm`; install project
requirements before running it. ImageNet weights are loaded by default and may be
downloaded by torchvision on the first run. Add `--no-pretrained` to initialize
the backbone randomly when working offline. The script freezes the backbone,
trains only the final classification layer, selects the checkpoint by lowest
validation loss, then evaluates that checkpoint once on the test set. Checkpoints
are saved under `models/transfer_frozen/`; training history, per-class metrics,
and confusion matrix outputs are saved under `results/transfer_frozen/`.
The Xception run report is available at
`results/transfer_frozen/xception/summary.md`.

The current aspect-aware frozen Xception experiment uses square crops, class/aspect-aware training sampling, and a rebalanced development split. It selects its checkpoint on validation only and does not load test images/features. From the project root, create the split and run it with:

```powershell
python -m src.create_aspect_split --output data/split_aspect.csv
python -m src.xception_aspect_frozen --split-csv data/split_aspect.csv --crop-mode square --aspect-balance --image-size 224 --epochs 30 --batch-size 32 --cpu-threads 4
```

The run reached 89.88% validation accuracy and 89.84% validation macro F1 at epoch 28. Its validation contains only four wide `smokefire` images, so it is not enough to estimate wide-camera performance reliably. Full results and limitations are documented in [`frozen_features.md`](frozen_features.md), with run artifacts under `results/transfer_frozen/xception_square_aspect_balanced_224/`.

The earlier 64/224/256/299 letterbox experiments were evaluated on the same original test set and were used to compare input sizes. Keep their metrics as historical diagnostics, not as an unbiased basis for selecting a new model. The aspect groups in that test set are also strongly correlated with class.

To run the aspect-robust Xception experiment as a separate branch from the
unchanged ResNet pipeline:

```powershell
python -m src.xception_aspect_frozen --epochs 30 --batch-size 32
```

The pretrained Xception input is 299×299 by default. To compare another
supported scale while preserving the same split and model setup, pass
`--image-size 256` (also supports 64, 224, 320, and 384); each size writes separate
features, checkpoints, and reports.

This command reproduces the earlier letterbox branch: it reads original images,
applies train-only crop/affine/flip/color augmentation, and caches frozen
features under `data/cache/`. Those historical reports include metrics by raw
aspect-ratio group, but their test results have already informed comparisons.
For new model selection, use the square-crop validation-only command above.
The completed experiment report is at
`results/transfer_frozen/xception_aspect_robust/summary.md`.
The 256×256 comparison is at
`results/transfer_frozen/xception_aspect_robust_256/summary.md`.
The 64×64 comparison is at
`results/transfer_frozen/xception_aspect_robust_64/summary.md`.
The 224×224 comparison is at
`results/transfer_frozen/xception_aspect_robust_224/summary.md`.

Fine-tune the final ResNet-50 block from the best frozen checkpoint with:

```bash
python -m src.fine_tune_transfer --epochs 8 --batch-size 32
```

Earlier backbone blocks and BatchNorm running statistics remain frozen. The
checkpoint is selected by validation macro F1; the final test report is written
under `results/transfer_finetuned/resnet50/`.

An experimental two-output fire/smoke classifier can be compared with the
four-class softmax baseline using:

```bash
python -m src.frozen_multilabel --epochs 20 --batch-size 32
```

### 6. Compare the results

Run:

```text
main.ipynb
```

to summarize and compare all model results.

---

## Final Comparison

The final results will be summarized using a table similar to:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Basic Neural Network | TBD | TBD | TBD | TBD |
| Custom Complex CNN | TBD | TBD | TBD | TBD |
| Transfer Learning – Frozen | TBD | TBD | TBD | TBD |
| Transfer Learning – Fine-tuned | TBD | TBD | TBD | TBD |

The final conclusion will be based on the actual experimental results.

---

## Notes

- The original dataset split is preserved for standard runs; the separate aspect-aware development split is available for the square-crop Xception run.
- Historical test metrics are not an independent final evaluation because multiple configurations were compared on the original test set.
- Data augmentation is applied only to training data.
- Frozen and Fine-tuned Transfer Learning use the same pretrained backbone.
- Fine-tuning starts from the best Frozen model.
- The square-crop frozen Xception protocol does not access test images/features during training or model selection. For an unbiased final evaluation, use a new external test source and score it once after fixing the model and preprocessing.
- All models are evaluated using the same metrics.
