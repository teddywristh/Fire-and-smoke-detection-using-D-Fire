# Forest fire and smoke classification (Forest Fire C4)

Four-class image classification: `fire`, `nofire`, `smoke`, `smokefire` (4,800 images, 1,200 per class).
This branch reruns every model from the `basic-nn`, `custom-complex-cnn` and `transfer-learning` branches on
**one shared data pipeline**, and adds a fine-tuning step built on the motif of `transfer_learning_all.py`
(frozen ImageNet backbone, then a `GlobalAveragePooling -> Dense(256) -> Dropout(0.5)` head). One framework
(Keras 3 / TensorFlow), one split, one notebook per task.

## Layout

```text
config.py                          dataset path, class names, image size, seed
src/data.py                        the shared data pipeline; `python -m src.data` rebuilds the split
src/experiment.py                  training, evaluation, plots, leaderboard, ImageNet backbones
src/compare.py                     bootstrap intervals for every run (results/comparison.csv)
notebooks/01_basic_nn.ipynb        task 1: basic neural network
notebooks/02_custom_cnn.ipynb      task 2: custom residual + squeeze-and-excitation CNN
notebooks/03_frozen_backbone.ipynb task 3: frozen Xception and ResNet-50
notebooks/04_finetune_backbone.ipynb task 4: fine-tuned Xception and ResNet-50 (plan + runs)
notebooks/05_error_analysis.ipynb  where the models fail (feeds REPORT.md)
REPORT.md                          results and weaknesses
data/split_aspect.csv              the split every task uses
results/<run>/                     metrics.json, history.csv, val/test predictions, summary.png
results/leaderboard.csv            one row per run
results/comparison.csv             macro F1 with 95% intervals, and difference from frozen ResNet-50
results/analysis/                  tables and figures of notebook 05
```

Checkpoints are written to `models/<run>.weights.h5` (not tracked).

## Setup

1. Download the dataset (Kaggle `obulisainaren/forest-fire-c4`). By default the code expects it next to the
   repository, at `../dataset/Forect Fire/Forest Fire_Dataset` (the folder that holds `train/`, `val/`,
   `test/`). Otherwise copy `local_config.example.py` to `local_config.py` and set `DATASET_ROOT`, or set the
   `FOREST_FIRE_DATASET_ROOT` environment variable.
2. `pip install -r requirements.txt` (tested with TensorFlow 2.21, Keras 3.15, Python 3.11, RTX 3050 6 GB).
3. Run the notebooks in order. Notebooks 01-04 each append their runs to `results/leaderboard.csv`;
   notebook 05 analyses them. Headless:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
   ```

`data/split_aspect.csv` is committed; `python -m src.data` rebuilds it from the raw images and gives the
same file.

## The shared data pipeline

The raw images do not share one shape, and the shape follows the class, differently in each split. Aspect
group = width/height in 0.9-1.1 (`square`), 1.6-1.95 (`wide`), anything else (`other`):

| Class | train square / wide / other | val square / wide / other | test square / wide / other |
|---|---|---|---|
| fire | 800 / 0 / 0 | 200 / 0 / 0 | 200 / 0 / 0 |
| nofire | 781 / 10 / 9 | 196 / 2 / 2 | 200 / 0 / 0 |
| smoke | 31 / 241 / 528 | 8 / 60 / 132 | 0 / 103 / 97 |
| smokefire | 770 / 8 / 22 | 190 / 4 / 6 | **0 / 99 / 101** |

Training `smokefire` is almost all square, test `smokefire` is never square. Stretching the images to a
square (the old `direct_resize`) or padding them (letterbox) both leave the shape visible to the model,
which can then learn "non-square means smoke". The pipeline of the transfer-learning branch removes that
shortcut, and every task now uses it (`src/data.py`):

1. **Split** `data/split_aspect.csv`: the Kaggle test split is kept untouched. The Kaggle train+val pool is
   re-split 800/200 per class, stratified by aspect group, so validation holds a few wide `smokefire`
   images (4) instead of 1.
2. **Square crop from the raw image**, then bilinear resize to 224x224: a random crop on train (side 72-100%
   of the short side, random position), the center crop on val/test. No stretching, no padding.
3. **Train-only augmentation**: small affine (rotate +-5 deg, translate +-4%, scale 0.94-1.06, shear +-2
   deg), horizontal flip, brightness and contrast x0.88-1.12.
4. **Aspect-balanced sampling**: every epoch draws 3,200 training images with replacement, with probability
   proportional to 1/sqrt(size of the image's (class, aspect group)), so the 8 wide training `smokefire`
   images are not drowned by the 770 square ones.

Batches are float32 pixels in [0, 255] with one-hot labels; each model rescales them itself ([0, 1] for the
custom networks, [-1, 1] for Xception, caffe BGR mean subtraction for ResNet-50). Every run picks its best
epoch on validation; test is predicted once, after training.

## The four tasks

Each notebook keeps the model and training recipe of the branch it comes from; only the data changed.

| Notebook | Source | Model and training (kept) |
|---|---|---|
| `01_basic_nn` | `basic-nn`, `src/basic_nn.py` | resize to 64x64, Flatten, Dense(256), Dropout(0.5), Dense(4); Adam 5e-4; 30 epochs; ReduceLROnPlateau; early stopping; best val loss |
| `02_custom_cnn` | `custom-complex-cnn`, `02_custom_cnn.ipynb` | stride-2 stem, 5 residual squeeze-and-excitation blocks, GAP, Dense(128); label smoothing 0.05; Adam 1e-3; 50 epochs; best val loss |
| `03_frozen_backbone` | `transfer-learning`, `xception_aspect_frozen.py` protocol | frozen ImageNet Xception / ResNet-50, head Dropout(0.45), Dense(512), BatchNorm, Dropout(0.2), Dense(4); AdamW 5e-4; 30 epochs; best val macro F1 |
| `04_finetune_backbone` | `transfer_learning_all.py` motif | stage 1 = the motif (frozen backbone, Dense(256), Dropout(0.5), Adam 1e-3, 10 epochs); stage 2 unfreezes the last stage (`top`) or every layer (`all`) at Adam 1e-5 for up to 20 epochs, BatchNorm frozen; best val macro F1 |

The fine-tuning plan and its reasons are written at the top of `04_finetune_backbone.ipynb`.

Two changes come from moving everything to one framework. The `transfer-learning` branch ran PyTorch;
here Xception is the Keras ImageNet checkpoint (the same weights timm's `legacy_xception` was converted
from), and ResNet-50 is the Keras ImageNet checkpoint (the original caffe weights, about 75% ImageNet
top-1) instead of torchvision's `DEFAULT` weights (about 81%). The frozen backbone also sees freshly
augmented images every epoch instead of one cached augmented view. As a check, frozen Xception reaches
89.9% validation macro F1 here against 89.84% for the PyTorch run on the same split.

## Results

One run per configuration (seed 42) on an RTX 3050 6 GB. Run times: notebook 01 2 min, 02 43 min, 03 13 min,
04 45 min. Intervals are 95% paired bootstrap intervals over the 800 val or 800 test images
(`python -m src.compare`, written to `results/comparison.csv`). Full rows: `results/leaderboard.csv`.
**These intervals understate test uncertainty:** test `smoke` / `smokefire` images are copies of a few
dozen photos, and resampling those groups instead of single images widens a model's test interval to
about +-4-7 points ([REPORT.md](REPORT.md), section 3).

| Run | Trainable params | Best / last epoch | Val acc | Val macro F1 | Val F1 vs frozen ResNet-50 | Test acc | Test macro F1 | Test F1 vs frozen ResNet-50 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 1 Basic NN | 3.15 M | 28 / 30 | 68.6 | 67.9 | -26.7 [-30.0, -23.6] | 60.5 | 60.9 | -17.5 [-21.2, -13.9] |
| 2 Custom CNN | 2.74 M | 46 / 50 | 95.6 | 95.6 | +1.0 [-0.7, +2.8] | 76.2 | 76.5 | -1.9 [-4.4, +0.4] |
| 3 Frozen Xception | 1.05 M | 15 / 21 | 89.9 | 89.9 | -4.7 [-6.9, -2.6] | 73.9 | 74.6 | -3.8 [-6.2, -1.4] |
| 3 Frozen ResNet-50 | 1.05 M | 11 / 17 | 94.6 | 94.6 | reference | 77.8 | 78.4 | reference |
| 4 Xception, stage 1 (motif, frozen) | 0.53 M | 6 / 10 | 88.8 | 88.8 | -5.8 [-8.1, -3.6] | 71.4 | 72.0 | -6.4 [-9.0, -3.9] |
| 4 Xception, stage 2 `top` | 7.30 M | 19 / 20 | 93.8 | 93.8 | -0.9 [-2.7, +1.0] | 75.9 | 76.6 | -1.8 [-3.8, +0.2] |
| 4 Xception, stage 2 `all` | 21.28 M | 19 / 20 | 96.9 | 96.9 | +2.3 [+0.6, +4.0] | 77.6 | 78.3 | -0.1 [-2.2, +2.0] |
| 4 ResNet-50, stage 1 (motif, frozen) | 0.53 M | 8 / 10 | 94.6 | 94.6 | 0.0 [-1.5, +1.4] | 78.0 | 78.6 | +0.2 [-1.2, +1.6] |
| 4 ResNet-50, stage 2 `top` | 15.48 M | 7 / 12 | 97.0 | 97.0 | +2.4 [+1.1, +3.8] | 77.5 | 78.1 | -0.3 [-2.1, +1.5] |
| 4 ResNet-50, stage 2 `all` | 24.01 M | 5 / 10 | **97.5** | **97.5** | +2.9 [+1.5, +4.3] | 77.9 | **78.6** | +0.2 [-1.6, +2.0] |

Test F1 per class:

| Run | fire | nofire | smoke | smokefire |
|---|---:|---:|---:|---:|
| 1 Basic NN | 71.2 | 66.1 | 59.8 | 46.4 |
| 2 Custom CNN | 74.7 | 92.7 | 70.1 | 68.5 |
| 3 Frozen Xception | 74.4 | 94.7 | 68.7 | 60.7 |
| 3 Frozen ResNet-50 | 77.5 | 98.0 | 73.4 | 64.7 |
| 4 Xception, stage 2 `all` | 76.2 | 99.2 | 71.5 | 66.1 |
| 4 ResNet-50, stage 2 `all` | 78.2 | 99.2 | 72.1 | 64.8 |

### What the numbers say

- **Validation separates the models; test does not.** Part of the reason is that validation holds
  augmented copies of the training photos: nearest-neighbour lookup alone scores 97.0% on it
  ([REPORT.md](REPORT.md), section 2). On validation, fine-tuning beats the frozen
  ResNet-50 by 2.3-2.9 points (`all` for both backbones, `top` for ResNet-50; intervals above zero). On
  test, the custom CNN, the frozen ResNet-50 and every stage-2 run land between 76.5 and 78.6 macro F1,
  and none of them is distinguishable from the frozen ResNet-50. Only the basic NN and the two frozen
  Xception runs are clearly worse.
- **Fine-tuning helps Xception more than ResNet-50.** Xception goes from 88.8 (motif, frozen) to 96.9
  validation F1 and from 72.0 to 78.3 test F1. ResNet-50's frozen features are already at 94.6 / 78.6, and
  fine-tuning moves validation to 97.5 but leaves test flat. Unfreezing everything scores at least as well
  as unfreezing the last stage and takes 1.5-1.8x the time per epoch.
- **The custom CNN trained from scratch is as good as the frozen ImageNet backbones on validation** (95.6),
  and reaches 76.5 on test, 1.9 points under the frozen ResNet-50 (the interval includes zero).
- **The 19-20 point gap between validation and test is the same for every strong model**, so it comes from
  the test split, not from the models. The errors that grow on test:

  | Error (images) | val, strong models | test, strong models |
  |---|---:|---:|
  | `fire` predicted `smokefire` (of 200) | 9-35 | 64-82 |
  | `smoke` predicted `smokefire` (of 200) | 0-7 | 57-81 |
  | `smokefire` predicted `smoke` (of 200) | 0-8 | 19-61 |

  Test `fire` images are square like the training ones, so the first error is not about shape. The eight
  test `fire` images the best run calls `smokefire` with the highest confidence all show a large smoke
  plume next to the flames, which matches the `smokefire` definition, while the training `fire` images
  we looked at show flames with little or no smoke. This suggests that part of the gap comes from the
  Kaggle test split labelling smoky fires differently from the training split. If a label audit confirms
  it (next steps), no model or pipeline change can close that part.
- **What the shared pipeline changed on test.** The test images are the same 800 as in the older
  branches. The old frozen ResNet-50 (PyTorch, stretched images) scored 78.4 test macro F1 with 77
  `smokefire -> smoke` errors and 70 `fire -> smokefire` errors. The frozen ResNet-50 here also scores
  78.4, with 44 and 68: the error that the image shape causes drops by about 40%, the label-related one
  stays. The basic NN (60.9 vs 60.1 before) and the custom CNN (76.5 vs 78.8) did not gain on test; the
  custom CNN previously trained on stretched images without augmentation.

### Caveats

- One seed per run, and GPU training is not bit-reproducible. Differences inside the intervals above
  (about +-1.5 points on validation, +-1.5-2.5 on test, and wider on test once the copies of one photo
  are resampled together) should not be read as rankings.
- Several runs stopped at or next to their epoch budget (basic NN 28/30, custom CNN 46/50, Xception
  fine-tuning 19/20), so they may not have converged. The budgets are the ones of the original recipes and
  of the plan, and were not changed after seeing results.
- Only validation was used to pick epochs and settings. The Kaggle test split had already been looked at
  in the older branches, so it is not an untouched holdout; read it as a check under distribution shift.
- Validation holds only 4 wide `smokefire` images, so it cannot measure the wide-camera failure.
- ResNet-50 uses the Keras (caffe) ImageNet weights, not the stronger torchvision ones of the older branch.

## Weaknesses and next steps

[REPORT.md](REPORT.md) analyses where the models fail. In short: validation contains augmented copies
of training photos, so it cannot rank the strong models; test `smoke` / `smokefire` come from a different,
higher-resolution source with few original photos; most shared test errors sit on the `fire` /
`smokefire` boundary; and the models are confidently wrong. Next steps, in order: split by source photo,
audit the `fire` / `smokefire` labels, report test per source, add independent data, calibrate, and only
then compare seeds.
