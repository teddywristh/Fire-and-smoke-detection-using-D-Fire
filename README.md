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

A pretrained CNN such as **EfficientNetB0** or **MobileNetV2** is used as a feature extractor.

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

### 3. Download the dataset

Download the dataset from Kaggle and place it inside:

```text
data/raw/
```

### 4. Run the experiments

Run the notebooks:

```text
01_basic_nn.ipynb
02_custom_cnn.ipynb
03_transfer_frozen.ipynb
04_transfer_finetuning.ipynb
```

The Fine-tuning experiment should start from the best Frozen Transfer Learning model.

### 5. Compare the results

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