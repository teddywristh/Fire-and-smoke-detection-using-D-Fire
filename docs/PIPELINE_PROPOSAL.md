# Proposal: one shared training and evaluation pipeline

Status: proposal for team review · 2026-10-06

## Summary

The `basic-nn` and `custom-complex-cnn` branches are now merged into `main`.
Both models use the shared data manifest correctly. Everything after loading
the data differs between them: label format, augmentation, callbacks,
metric names, result files and where the code lives. As a result their
outputs cannot go into one comparison table without manual work, and the
Transfer Learning branches have no template to follow.

This proposal moves the common steps (data → augmentation → training →
evaluation → reporting) into shared modules under `src/`. Each model branch
then adds just one model file, one notebook and its results folder.

There is also one finding about the data that affects every model: **the
validation set is not representative of the test set for `smoke` and
`smokefire`**. Section 3 explains it. It changes how we should report results.

The team needs to decide five points (Section 8).

## 1. Current results

| Model | Input | Params | Val accuracy | Test accuracy | Test macro F1 | fire→nofire | smokefire→nofire |
|---|---|---:|---:|---:|---:|---:|---:|
| Basic NN | 64×64 | 3.15M | 0.7250 | 0.5975 | 0.6006 | 7 | 17 |
| Custom Complex CNN | 224×224 | 2.74M | 0.9563 | 0.7825 | 0.7875 | 6 | 0 |

Val accuracy is taken at the checkpoint epoch (minimum `val_loss`). The Custom
CNN values were re-checked on 2026-10-06 by loading `models/custom_cnn_best.keras`
and predicting val and test, and the test numbers matched `metrics.json`
exactly. The Basic NN checkpoint is not in the repository, so its numbers
come from `results/basic_nn/` and were not re-checked.

## 2. Review of the two branches

Requirements come from the project `README.md` (Training Protocol, Evaluation
Metrics, Error Analysis) and `DATA_PROCESSING.md`.

| Requirement | Basic NN | Custom CNN |
|---|---|---|
| Reads `data/split.csv` and processed images | Yes | Yes |
| Keeps the original split | Yes | Yes |
| Augmentation on train only | Yes: flip, brightness, contrast | **No augmentation at all**, although the design doc says otherwise |
| Validation used for checkpoint and early stopping | Yes (`val_loss`) | Yes (`val_loss`) |
| Test used only for final evaluation | Earlier configurations were picked by test accuracy (disclosed in `basic_nn_results.md`). The current tuning round selects by validation | Yes |
| Fixed random seed | `tf.keras.utils.set_random_seed(42)` | Shuffle seed only, so weight init and dropout are not seeded |
| Accuracy, precision, recall, F1 saved | Yes, `classification_report.csv` | Printed in the notebook, not saved |
| Confusion matrix saved | CSV and PNG | PNG only |
| Train/val curves | Yes | Yes |
| Critical false negatives | Counted in `basic_nn_results.md` | `critical_fire_as_nofire.png` shows 1 image while the confusion matrix has 6, and no notebook cell creates it |
| Misclassified images with true/pred/confidence | `misclassified.csv` and a 16-image grid | Missing; test predictions were not saved |
| Reproducible from the repository | The tuning trials behind `tuning_summary.json` were run by code that is not committed | Several cells were edited after the run (the code prints `CPUs available` but the saved output says `GPUs available`; some cells have output but no execution count) |
| Shared files left untouched | Rewrote `README.md` and `DATA_PROCESSING.md` | Changed `data/split.csv` to a machine-specific path, dropped `models/*` from `.gitignore`, committed a 33 MB checkpoint |

What was changed while merging:

- `README.md` and `DATA_PROCESSING.md` restored to the shared versions. The
  Basic NN docs remain in `results/basic_nn/`.
- The Custom CNN notebook moved to `notebooks/02_custom_cnn.ipynb`. Its design
  notes moved to `results/custom_cnn/custom_cnn_method.md`, with an
  integration note listing where the run differs from the design.
- The `data/split.csv`, `.gitignore` and checkpoint changes were left out.
  The branch was squash-merged so the 33 MB file does not enter `main`'s
  history. The checkpoint is unchanged and still exists outside Git.
- `.gitignore` now ignores `data/raw/*`. This prevents the problem the Custom
  CNN branch hit when it stored the dataset inside the project.

Smaller issues, not fixed:

- `basic_nn_method.md` links to slide PDFs at `../../../Chapter*.pdf`. Those
  files are outside the repository, so the links are broken for everyone else.
- Custom CNN trained on a Windows CPU at 7–16 minutes per epoch. TensorFlow
  ≥ 2.11 has no GPU support on native Windows, so WSL2 is the way to get one.

## 3. Data finding: validation does not predict test for smoke and smokefire

Both models lose 13–17 points from validation to test. For Custom CNN the
loss is concentrated in three cells of the confusion matrix:

| Custom CNN error | Validation (of 200) | Test (of 200) |
|---|---:|---:|
| fire → smokefire | 14 | 68 |
| smoke → smokefire | 0 | 53 |
| smokefire → smoke | 1 | 37 |
| nofire correct | 194 | 192 |

The raw image sizes in `data/split.csv` suggest why. For `smoke` and
`smokefire`, test images come from a different source than train and val:

| Class | Train | Validation | Test |
|---|---|---|---|
| smoke | Small web images, median width 299, aspect ratio 1.50, 52 distinct sizes | Same profile. All 50 sizes also occur in train | Large frames (852×480, 2880×1620, 1440×1080), median aspect ratio 1.78. Only 3 of 8 sizes occur in train |
| smokefire | 766/800 images are 250×250 | 194/200 are 250×250 | **0/200** are 250×250. Median aspect ratio 1.78 |
| fire | 800/800 are 250×250 | 200/200 | 200/200 |

What this means:

1. Validation results overstate how well a model handles `smoke` and
   `smokefire`. A model that is best on validation is not necessarily best on
   test. The final comparison must show validation and test side by side.
2. In the training data, image geometry is tied to class: `fire` and
   `smokefire` are mostly 250×250 squares, while `smoke` is wide. Direct resize
   to 224×224 squeezes wide images and leaves a visible signature. Test
   `smokefire` images are wide, so they may look like the `smoke` source to a
   model. This fits Basic NN predicting `smokefire → smoke` for 98 of 200 test
   images, but it is **a hypothesis, not a proven cause**.
3. Exact duplicates were already ruled out (SHA256). A 256-bit dHash comparison
   on 2026-10-06 found no train image within distance 20 of any val or test
   image. dHash does not survive rotation or shifting, though. The Kaggle
   README says the images were made with `ImageDataGenerator` augmentation, so
   augmented copies across splits are still possible. An embedding-based
   similarity check would settle it.

What this does **not** change: the README requires keeping the original split,
and this proposal keeps it. The fix is in reporting and augmentation, not
re-splitting.

Cheap checks to add to the shared pipeline:

- Save a validation confusion matrix next to the test one for every model.
- Try aspect-ratio jitter or random-resized-crop in train augmentation, and
  see whether test `smoke`/`smokefire` recall improves while validation stays
  the same. Choose the policy by validation; report test once.

## 4. Where the branches differ, and the proposed standard

| Aspect | Basic NN | Custom CNN | Proposed standard |
|---|---|---|---|
| Code location | `src/basic_nn.py` script | All code inside the notebook | Model in `src/models/<model>.py`; shared steps in `src/`; notebook calls them |
| Dataset input | Resized to 64×64 in `tf.data` | 224×224 | Same dataset for all models: 224×224, float32 in [0, 255]. The model does its own resizing and scaling as its first layers |
| Pixel scaling | ÷255 in `tf.data` | ÷255 in `tf.data` | Inside the model (`Rescaling`, or the backbone's own preprocessing), so the saved `.keras` file works on raw images for demo and Grad-CAM |
| Labels / loss | Integer labels, sparse CCE | One-hot, CCE with label smoothing 0.05 | One-hot, `CategoricalCrossentropy(label_smoothing=…)`, default 0 |
| Augmentation | Flip, brightness ±0.08, contrast 0.9–1.1 | None | One policy in `src/augmentation.py`, train only, the same for every model |
| Seed | Global | Shuffle only | `tf.keras.utils.set_random_seed(config.RANDOM_SEED)` in the shared trainer |
| Optimizer, LR, epochs | Adam 5e-4, 30 | Adam 1e-3, 50 | Per model, stored in the model spec and saved to `run_config.json` |
| EarlyStopping | `val_loss`, patience 5; best model reloaded from checkpoint | `val_loss`, patience 12, `restore_best_weights` | `val_loss`, `restore_best_weights=True`; patience per model |
| ReduceLROnPlateau | factor 0.5, patience 2, min 1e-5 | factor 0.5, patience 4, min 1e-6 | Shared defaults (factor 0.5, patience 3, min 1e-6), overridable per model |
| History file | `training_history.csv` | `training_log.csv` | `training_history.csv` (from `CSVLogger`) |
| `metrics.json` keys | `test_accuracy`, `macro_f1`, `parameter_count`, … | `test_accuracy`, `test_macro_f1`, `total_parameters`, … | One schema (Section 5.3) |
| Checkpoint in Git | No | Yes, 33 MB | No. Share the file outside Git and record its SHA256 |

Input size and architecture can differ per model. That is the point of the
comparison. Everything else should be the same, so that differences in the
results come from the model.

## 5. Proposed shared pipeline

### 5.1 Layout

```text
src/
├── data_loader.py      # existing: builds data/split.csv and processed images
├── datasets.py         # new: manifest → tf.data for train/val/test
├── augmentation.py     # existing policy, implemented as Keras layers
├── training.py         # new: seed, compile, callbacks, fit, history
├── evaluation.py       # metrics, reports, predictions, critical errors
├── visualization.py    # curves, confusion matrices, error grids
├── gradcam.py          # final model only
├── compare.py          # new: results/*/metrics.json → results/final_comparison.csv
├── run.py              # new: python -m src.run --model <name> [--evaluate-only]
└── models/
    ├── __init__.py     # registry: name → ModelSpec
    ├── basic_nn.py     # one file per model branch
    ├── custom_cnn.py
    ├── transfer_frozen.py
    └── transfer_finetuned.py
notebooks/
└── 0X_<model>.ipynb    # calls src.run; shows figures; holds discussion
results/<model>/        # fixed set of files (5.3)
```

### 5.2 What a model branch writes

A model file only describes the model and its training settings:

```python
# src/models/custom_cnn.py
import tensorflow as tf
from src.models import ModelSpec


def build(spec: ModelSpec) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(224, 224, 3))   # every dataset yields 224×224, [0, 255]
    x = tf.keras.layers.Rescaling(1 / 255)(inputs)
    ...                                             # stem, residual-SE blocks, head
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=spec.name)


SPEC = ModelSpec(
    name="custom_cnn",
    display_name="Custom Complex CNN",
    build=build,
    learning_rate=1e-3,
    epochs=50,
    early_stopping_patience=12,
    label_smoothing=0.05,
)
```

Basic NN would start with `Resizing(64, 64)` and then `Rescaling(1 / 255)`.
Transfer models would use the backbone's own preprocessing: Keras EfficientNet
models scale inputs internally and expect [0, 255], while MobileNetV2 needs
`mobilenet_v2.preprocess_input`.

All model types go through one shared entry point:

```bash
python -m src.run --model custom_cnn                # train, then evaluate
python -m src.run --model custom_cnn --evaluate-only
python -m src.compare                               # build final_comparison.csv
```

### 5.3 Result contract

Every `results/<model>/` contains the same files, all written by shared code:

| File | Content |
|---|---|
| `run_config.json` | Spec values, git commit, TensorFlow version, date, device, checkpoint SHA256 |
| `training_history.csv` | Per-epoch loss, accuracy, val_loss, val_accuracy, learning_rate |
| `metrics.json` | Summary in the schema below |
| `classification_report.csv` | Per-class precision, recall, F1 on test |
| `confusion_matrix.csv`, `confusion_matrix_val.csv` | Counts; rows = true label |
| `predictions.csv` | One row per test image: path, true label, predicted label, confidence, four class probabilities |
| `misclassified.csv`, `critical_errors.csv` | Subsets of `predictions.csv` |
| `learning_curves.png`, `confusion_matrix.png`, `misclassified_samples.png`, `critical_errors.png` | Figures |
| `<model>_method.md` | Hand-written: architecture, choices, deviations |
| `<model>_results.md` | Discussion of the numbers above |

`metrics.json` schema:

```json
{
  "model": "custom_cnn",
  "display_name": "Custom Complex CNN",
  "input_size": [224, 224],
  "parameter_count": 2742628,
  "epochs_trained": 50,
  "best_epoch": 50,
  "selection_metric": "val_loss",
  "val":  {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0},
  "test": {"loss": 0.0, "accuracy": 0.0, "macro_precision": 0.0,
           "macro_recall": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0},
  "critical_errors": {"fire_as_nofire": 0, "smokefire_as_nofire": 0,
                      "fire_smokefire_confusions": 0, "smoke_smokefire_confusions": 0}
}
```

`src/compare.py` reads these files and writes `results/final_comparison.csv`
with validation and test columns side by side (see Section 3).

### 5.4 Make the manifest machine-independent

`data/split.csv` stores `filepath` relative to the project. Running
`python -m src.data_loader` with the dataset in another place therefore
rewrites all 4,800 rows, as happened on the Custom CNN branch. Proposed fix:
store `filepath` relative to `DATASET_ROOT` (`train/fire/fire_train_1001.jpg`),
leave the dataset root out of `processing_summary.json` and
`processing_report.md`, and write the CSV with `lineterminator="\n"`. After that, the file is byte-identical on every
machine and OS.

## 6. Branch rules

A model branch may change only:

- `src/models/<model>.py`
- `notebooks/0X_<model>.ipynb`
- `results/<model>/`
- `requirements.txt`, with a note in the commit message

Every other file is shared. That includes `README.md`, `DATA_PROCESSING.md`,
`config.py`, the shared `src/` modules, `data/split.csv`, `data/processed/*`
and `.gitignore`. Changes to shared files go in a separate branch that the
team reviews.

Before committing:

1. Run *Restart & Run All* on the notebook, so outputs match the code.
2. Commit no checkpoints, datasets or `local_config.py`.
3. Check `git diff --stat main` and confirm that only your own paths appear.
4. Select on validation only. Run the test set once, at the end.

## 7. Migration plan

| Step | Owner | Work |
|---|---|---|
| 1 | One person, shared branch | Implement 5.1–5.4: `datasets.py`, `augmentation.py`, `training.py`, `evaluation.py`, `visualization.py`, `run.py`, `compare.py`, the model registry, the manifest fix |
| 2 | Basic NN owner | Port `src/basic_nn.py` to `src/models/basic_nn.py`. Re-run with the shared pipeline. Keep `experiment_comparison.csv` as history. Commit or remove the tuning code behind `tuning_summary.json` |
| 3 | Custom CNN owner | Port the notebook model to `src/models/custom_cnn.py`. Re-run with the shared augmentation (the current run has none). Turn the notebook into a thin driver. Train on WSL2 or a GPU machine if possible |
| 4 | Transfer Learning owners | Start directly on the standard: `transfer_frozen.py`, then `transfer_finetuned.py` loading the best frozen checkpoint |
| 5 | Anyone | `main.ipynb` reads `results/final_comparison.csv`; Grad-CAM on the fine-tuned model |

The current results of Steps 2 and 3 stay valid until re-run. After the
re-run, any difference in numbers should be explained by the pipeline change
(augmentation, seed), and noted in `<model>_results.md`.

## 8. Decisions needed from the team

1. **Input contract.** Recommended: one dataset for all models (224×224,
   [0, 255]), with resizing and scaling inside each model. The alternative is a
   separate dataset per model.
2. **Augmentation.** Recommended: one shared policy for every model, so the
   comparison is fair. Should it include aspect-ratio jitter (Section 3)?
3. **Entry point.** Recommended: `python -m src.run` as the source of truth,
   with notebooks for presentation. The alternative is notebook-only.
4. **Checkpoint sharing.** Google Drive folder or GitHub Release assets, with
   the SHA256 recorded in `run_config.json`.
5. **Documentation language.** The shared docs are in English, while the model
   docs are in Vietnamese. Pick one language for the final report.
