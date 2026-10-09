# Basic Neural Network — Phân loại cháy rừng và khói

Hiện tại dự án chỉ tập trung phát triển **Basic NN dùng Dense**.
Toàn bộ code nằm trong notebook, ghi chú tiếng Việt có dấu rồi đến code.
Không có class tự viết; notebook mô hình chỉ có một hàm đọc ảnh.

## Cách chạy

Cài thư viện: `python -m pip install -r requirements.txt`.
Chọn kernel Python có các thư viện đó, chạy cell từ trên xuống.

1. [00 — Chuẩn bị dữ liệu](notebooks/00_data_preparation.ipynb): chỉ cần khi
   chưa có manifest/ảnh processed hoặc đổi dữ liệu. Sửa `DATASET_ROOT` ở cell đầu.
2. [01 — Basic NN](notebooks/01_basic_nn.ipynb): sáu bước import → dữ liệu
   → model/compile → train → evaluate → báo cáo.

Notebook 01 mặc định `TRAIN_MODEL=True`. Đổi False để đánh giá checkpoint đã lưu.
Baseline cũ được giữ ở `models/basic_nn_best.keras` và `results/basic_nn/`.
Bản mới lưu riêng ở `models/basic_nn_notebook.keras` và `results/basic_nn_notebook/`.

## Dữ liệu và mô hình

Mapping: fire=0, nofire=1, smoke=2, smokefire=3.
Giữ nguyên 3.200 train, 800 validation, 800 test; bốn lớp cân bằng.
Ảnh RGB 64×64, pixel [0, 1]. Train shuffle và lật ngang; val/test giữ thứ tự.

Basic NN mới thử Dense(256) → BatchNormalization → ReLU → Dropout(0,3)
→ Dense(128, ReLU) → Dropout(0,2) → Softmax(4).
Adam 0,0003, tối đa 30 epoch, checkpoint chọn bằng validation loss.
Nhãn số nguyên dùng sparse cross-entropy, không cần one-hot.

## Kết quả và giới hạn

Baseline cũ đạt 59,75%. Kết quả bản mới được đo thực tế trong notebook 01 và
`results/basic_nn_notebook/`, gồm history, metrics, predictions, F1,
confusion matrix và biểu đồ. Không dùng test để chọn epoch hoặc tuning tiếp.

Bản Dense mới đạt **60,25%**, tăng 0,5 điểm phần trăm; Macro F1 giảm từ
0,6006 xuống 0,5928. Đây là thay đổi nhỏ của một seed, chưa chứng minh cải thiện
ổn định. Xem [kết quả Basic NN](results/basic_nn_notebook/basic_nn_results.md).

[Rà soát nhãn](results/basic_nn/label_audit.md) có các mẫu raw kéo sọc/méo
và nhãn cần duyệt lại. `data/label_review_candidates.csv` chỉ hỗ trợ rà nhãn,
không tự đổi nhãn hoặc loại ảnh test khó. Mạng Dense không thể khôi phục
ảnh hỏng hoặc tự xác nhận nhãn đúng. Test đã được dùng trong lựa chọn lịch sử;
cần holdout mới để đánh giá độc lập.

---

## Cập nhật từ main và hướng dẫn chung của nhóm

> Nhánh `basic-nn` đã nhận cập nhật từ `main`. Phần Basic NN đang phát triển
> vẫn chạy trực tiếp trong notebook 01. Các module `src/` và `config.py` từ
> main được giữ cho pipeline chung của nhóm; notebook 01 không import chúng.
> Kết quả Basic NN hiện tại lưu riêng trong `results/basic_nn_notebook/`.

# Comparative Deep Learning Models for Forest Fire and Smoke Image Classification

## Project Overview

This project develops and compares Deep Learning models for forest fire and smoke image classification.

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

The original split is preserved so that all models are trained and evaluated using the same data.

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

A pretrained CNN is used as a feature extractor with a newly initialized classification head. The current implementation (PyTorch, `src/training/`) supports **ResNet-50**, **EfficientNet-B0**, **MobileNetV3-Large**, and **Xception**.

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

- All models use the same train, validation, and test split.
- Data augmentation is applied only to training data.
- Validation data is used for model selection and Early Stopping.
- Test data is used only for final evaluation.
- A fixed random seed is used where possible.
- The same evaluation metrics are used for all models.

Exception: several historical frozen-transfer configurations were compared on
the original test set, so those test scores are exploratory rather than
independent estimates. The aspect-aware Xception experiment uses a separate
development split (`data/split_aspect.csv`) that repartitions only train and
validation; test membership is unchanged. See `docs/frozen_features.md`.

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
│   ├── data_loader.py        # shared: builds split.csv and processed images
│   ├── preprocessing.py      # shared: image checks and direct resize
│   ├── datasets.py           # shared: split.csv -> tf.data for every model
│   ├── augmentation.py       # shared: train-only augmentation policy
│   ├── models/               # one Keras file per model, owned by one person
│   │   ├── basic_nn.py
│   │   ├── custom_cnn.py
│   │   └── custom_cnn_new.py
│   ├── data/                 # Kaggle download, aspect-aware split; data_loader/preprocessing re-export the shared ones
│   │   ├── download_dataset.py
│   │   ├── create_aspect_split.py
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── training/             # PyTorch transfer learning (frozen, fine-tuned, variants)
│   │   ├── frozen_transfer.py
│   │   ├── fine_tune_transfer.py
│   │   ├── xception_aspect_frozen.py
│   │   ├── frozen_multilabel.py
│   │   └── calibrate_frozen.py
│   └── analysis/             # evaluation and plots for transfer runs; gradcam.py is a TODO
│       ├── evaluation.py
│       ├── visualization.py
│       ├── augmentation.py
│       └── gradcam.py
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
- PyTorch, torchvision, timm (transfer learning in `src/training/`)
- KaggleHub (optional dataset download)
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

Run the data preparation pipeline:

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

If the raw dataset is not on disk yet, `python -m src.data.download_dataset`
downloads it with KaggleHub into `data/raw/kagglehub/` and then runs the same
pipeline. `python -m src.data.data_loader` is an alias of `python -m src.data_loader`.

For full data-processing details, see:

```text
DATA_PROCESSING.md
```

### 5. Run the experiments

All models share one data pipeline (`src/datasets.py`): the same split, the
same 224×224 input with pixel values in [0, 255], one-hot labels, and the same
train-only augmentation (`src/augmentation.py`). Each model lives in its own
file under `src/models/`, which holds its architecture, training settings and
evaluation, and does its own resizing and scaling as its first layers. The
contract is described in `src/models/__init__.py`.

| Model | Run | Status |
|---|---|---|
| Basic Neural Network | `python -m src.models.basic_nn` | v1 results recorded; re-run needed on the shared pipeline |
| Custom Complex CNN | `python -m src.models.custom_cnn` | v1 results recorded (`notebooks/02_custom_cnn.ipynb`); re-run needed on the shared pipeline |
| Transfer Learning – Frozen | `python -m src.training.frozen_transfer` | PyTorch implementation; not on the shared pipeline yet (see below) |
| Transfer Learning – Fine-tuned | `python -m src.training.fine_tune_transfer` | PyTorch implementation; not on the shared pipeline yet (see below) |

Add `--evaluate-only` to evaluate a saved checkpoint without training. The
Fine-tuning experiment should start from the best Frozen Transfer Learning model.

#### Transfer learning (PyTorch)

The transfer-learning scripts in `src/training/` read `data/split.csv` and the
processed images, but they do not use `src/datasets.py` or
`src/augmentation.py`: they use PyTorch, ImageNet normalization and their own
train-time augmentation. Their results are therefore not directly comparable
with the Keras models above. Checkpoints go to `models/transfer_frozen/`;
reports go to `results/transfer_frozen/` and `results/transfer_finetuned/`.

```bash
# Frozen backbone, linear head; checkpoint chosen by validation loss
python -m src.training.frozen_transfer --architecture resnet50 --epochs 20 --batch-size 32
# Other backbones: efficientnet_b0, mobilenet_v3_large, xception (299x299, needs timm)

# Fine-tune ResNet-50 layer4 + head from the best frozen checkpoint
python -m src.training.fine_tune_transfer --epochs 8 --batch-size 32

# Variants: two-output fire/smoke head, and validation-based logit calibration
python -m src.training.frozen_multilabel --epochs 20 --batch-size 32
python -m src.training.calibrate_frozen

# Aspect-aware frozen Xception: square crop, class/aspect-balanced sampling,
# separate train/val split, no test evaluation
python -m src.data.create_aspect_split --output data/split_aspect.csv
python -m src.training.xception_aspect_frozen --split-csv data/split_aspect.csv --crop-mode square --aspect-balance --image-size 224 --epochs 30 --batch-size 32
```

ImageNet weights are downloaded on the first run; add `--no-pretrained` to work
offline. The Xception aspect experiment reads raw images from `DATASET_ROOT`
and caches backbone features in `data/cache/`. Protocols, the experiment history
and the reported metrics are in `docs/frozen_features.md`; per-pipeline
preprocessing is in `docs/data_preprocessing_ff.md`.

To add a model, create `src/models/<name>.py` that builds the model on
`src.datasets.INPUT_SHAPE`, trains with `src.datasets.load_datasets()` (or
`load_frames()` + `make_dataset()`), saves `models/<name>_best.keras` and writes
its outputs to `results/<name>/`. Use `src/models/custom_cnn.py` as an example.

Model checkpoints in `models/` are not tracked by Git. Background and next
steps are in `docs/PIPELINE_PROPOSAL.md`.

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

- The original dataset split is preserved.
- All models use the same train, validation, and test sets.
- Data augmentation is applied only to training data.
- Frozen and Fine-tuned Transfer Learning use the same pretrained backbone.
- Fine-tuning starts from the best Frozen model.
- The test set is used only for final evaluation.
- All models are evaluated using the same metrics.
