# Forest fire and smoke classification (Forest Fire C4)

Four-class image classification: `fire`, `nofire`, `smoke`, `smokefire` (4,800 images, 1,200 per class).
Four tasks, one self-contained notebook each, all on **the same data pipeline**: the models of the
`basic-nn`, `custom-complex-cnn` and `transfer-learning` branches, and a fine-tuning step built on the
motif of `transfer_learning_all.py`, a CIFAR-10 / VGG16 sample script that is not in this repository (frozen
ImageNet backbone under a `GlobalAveragePooling -> Dense(256) -> Dropout(0.5)` head). Keras 3 / TensorFlow.

## Layout

```text
notebooks/01_basic_nn.ipynb          task 1: basic neural network
notebooks/02_custom_cnn.ipynb        task 2: custom residual + squeeze-and-excitation CNN
notebooks/03_frozen_backbone.ipynb   task 3: frozen Xception and ResNet-50
notebooks/04_finetune_backbone.ipynb task 4: fine-tuned Xception and ResNet-50 (plan + runs)
results/<run>/                       metrics.json, history.csv, val/test predictions, summary.png
results/leaderboard.csv              one row per run
results/split_aspect.csv             the split every notebook rebuilt and used
```

Each run trained to the end leaves `models/<run>.weights.h5` (best epoch) and `models/<run>.fit.json` (its
history and `fit()` arguments); neither is tracked.

## Running

Each notebook is self-contained: it imports no project file and reads no project file, so any one of
them can be run alone, in any order. The four `.ipynb` files are therefore enough to hand over: copied into
any folder, they run there, and their first cell says how. Add a `data/` folder holding the dataset or its
`.zip` next to them only for a machine without internet, and the `models/` files (1.3 GB for all ten runs)
to skip the training.

1. `pip install -r requirements.txt` (tested with TensorFlow 2.21, Keras 3.15, Python 3.11, RTX 3050 6 GB).
2. The dataset is Kaggle `obulisainaren/forest-fire-c4` (version 2). Each notebook looks for the folder that
   holds `train/`, `val/` and `test/`, up to three levels down, in `DATASET_ROOT` (its first code cell) or
   `$FOREST_FIRE_DATASET_ROOT` when set, else in `data/`, `../dataset/` and `/kaggle/input`. It takes only a
   folder whose images are byte for byte the ones behind the results below (SHA-256 of every image), so a
   resized copy with the same file names is skipped. If none has them, it unpacks the `.zip` in `data/`,
   first downloading it from Kaggle's public endpoint when there is none (129 MB, no Kaggle account,
   standard library only). The same download by hand:

   ```bash
   curl -L -o data/forest-fire-c4-v2.zip \
     "https://www.kaggle.com/api/v1/datasets/download/obulisainaren/forest-fire-c4?datasetVersionNumber=2"
   ```

3. Run a notebook. It writes `results/` and `models/` (and `data/` when it downloads) next to the
   `notebooks/` folder, or next to the notebook itself when its folder has another name. With
   `USE_CHECKPOINTS = True` (the default, first code cell) a run that left both files in `models/` is
   reloaded and only scored, unless its `fit()` arguments changed; set it to `False` to retrain. Rescoring
   on a GPU can move one or two images whose top two classes nearly tie (up to 0.2 points). Headless:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
   ```

Every notebook rebuilds the split from the raw images and checks two fingerprints against the ones used for
the results below (file names per split, and the bytes of every image), so all four always train and score
on exactly the same images.

## The data pipeline (identical in all four notebooks)

The raw images do not share one shape, and the shape follows the class, differently in each split. Aspect
group = width/height in 0.9-1.1 (`square`), 1.6-1.95 (`wide`), anything else (`other`):

| Class | train square / wide / other | val square / wide / other | test square / wide / other |
|---|---|---|---|
| fire | 800 / 0 / 0 | 200 / 0 / 0 | 200 / 0 / 0 |
| nofire | 781 / 10 / 9 | 196 / 2 / 2 | 200 / 0 / 0 |
| smoke | 31 / 241 / 528 | 8 / 60 / 132 | 0 / 103 / 97 |
| smokefire | 770 / 8 / 22 | 190 / 4 / 6 | **0 / 99 / 101** |

Training `smokefire` is almost all square, test `smokefire` is never square. Stretching the images to a
square or padding them leaves the shape visible to the model, which can then learn "non-square means smoke".
The pipeline of the transfer-learning branch removes that shortcut:

1. **Split:** the Kaggle test split is kept untouched. The Kaggle train+val pool is re-split 800/200 per
   class, stratified by aspect group, so validation holds 4 wide `smokefire` images instead of 1.
2. **Square crop from the raw image**, then bilinear resize to 224x224: a random crop on train (side 72-100%
   of the short side, random position), the center crop on val/test. No stretching, no padding.
3. **Train-only augmentation:** small affine (rotate +-5 deg, translate +-4%, scale 0.94-1.06, shear +-2 deg),
   horizontal flip, brightness and contrast x0.88-1.12.
4. **Aspect-balanced sampling:** every epoch draws 3,200 training images with replacement, with probability
   proportional to 1/sqrt(size of the image's (class, aspect group)), so the 8 wide training `smokefire`
   images are not drowned by the 770 square ones.

Batches are float32 pixels in [0, 255] with one-hot labels; each model rescales them itself. Every run picks
its best epoch on validation; test is predicted once, after training.

## The four tasks

Each notebook keeps the model and training recipe of the latest commit of the branch it comes from; only the
data pipeline is shared.

| Notebook | Source | Model and training |
|---|---|---|
| `01_basic_nn` | `basic-nn` @ `21a47a8` | 64x64 (antialiased), ImageNet mean/std, random flip, Flatten, Dense(256, BN, ReLU), Dropout(0.3), Dense(128), Dropout(0.2), Dense(4); Adam 3e-4; 30 epochs; ReduceLROnPlateau; early stopping; best val loss |
| `02_custom_cnn` | `custom-complex-cnn` @ `c02b33a` | stride-2 stem, 5 residual squeeze-and-excitation blocks (ReLU), GAP, Dropout(0.4), Dense(128); Adam 1e-3; 50 epochs; ReduceLROnPlateau; early stopping (patience 10); best val loss |
| `03_frozen_backbone` | `transfer-learning`, `xception_aspect_frozen.py` protocol | frozen ImageNet Xception / ResNet-50, head Dropout(0.45), Dense(512), BatchNorm, Dropout(0.2), Dense(4); AdamW 5e-4; 30 epochs; best val macro F1 |
| `04_finetune_backbone` | `transfer_learning_all.py` motif | stage 1 = the motif (frozen backbone, Dense(256), Dropout(0.5), Adam 1e-3, 10 epochs); stage 2 unfreezes the last stage (`top`) or every layer (`all`) at Adam 1e-5 for up to 20 epochs, BatchNorm frozen; best val macro F1 |

The fine-tuning plan and its reasons are written at the top of `04_finetune_backbone.ipynb`. The
`transfer-learning` branch ran PyTorch; here Xception is the Keras ImageNet checkpoint (the weights timm's
`legacy_xception` was converted from) and ResNet-50 the Keras ImageNet checkpoint (the original caffe weights,
about 75% top-1) instead of torchvision's `DEFAULT` weights (about 81%).

## Results

One run per configuration (seed 42) on an RTX 3050 6 GB; training took 2, 34, 21 and 59 minutes per notebook.
Macro F1 and accuracy in %, from `results/leaderboard.csv`.

| Run | Trainable params | Best / last epoch | Minutes | Val acc | Val macro F1 | Test acc | Test macro F1 | Test F1 fire / nofire / smoke / smokefire |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 Basic NN | 3.18 M | 27 / 30 | 2 | 70.1 | 69.5 | 63.5 | 63.4 | 72.2 / 71.3 / 64.6 / 45.3 |
| 2 Custom CNN | 2.74 M | 46 / 50 | 34 | 95.1 | 95.1 | 76.6 | 77.0 | 75.0 / 94.5 / 70.7 / 67.9 |
| 3 Frozen Xception | 1.05 M | 22 / 28 | 13 | 90.8 | 90.7 | 74.9 | 75.4 | 73.7 / 96.5 / 73.2 / 58.3 |
| 3 Frozen ResNet-50 | 1.05 M | 10 / 16 | 8 | 94.1 | 94.1 | 79.0 | 79.6 | 79.2 / 97.7 / 73.7 / 67.9 |
| 4 Xception, stage 1 (motif, frozen) | 0.53 M | 9 / 10 | 4 | 90.2 | 90.3 | 73.8 | 74.0 | 72.7 / 94.9 / 71.7 / 56.6 |
| 4 Xception, stage 2 `top` | 7.30 M | 20 / 20 | 12 | 94.0 | 94.1 | 74.8 | 75.5 | 73.9 / 98.5 / 67.9 / 61.7 |
| 4 Xception, stage 2 `all` | 21.28 M | 10 / 15 | 15 | 96.2 | 96.3 | 76.4 | 77.1 | 76.5 / 99.0 / 70.5 / 62.6 |
| 4 ResNet-50, stage 1 (motif, frozen) | 0.53 M | 10 / 10 | 5 | 94.9 | 94.9 | 79.6 | 80.2 | 80.9 / 98.3 / 73.5 / 68.0 |
| 4 ResNet-50, stage 2 `top` | 15.48 M | 8 / 13 | 8 | 97.5 | 97.5 | 77.6 | 78.2 | 78.2 / 98.8 / 68.6 / 67.0 |
| 4 ResNet-50, stage 2 `all` | 24.01 M | 10 / 15 | 16 | **97.9** | **97.9** | 78.0 | 78.4 | 81.1 / 98.5 / 66.5 / 67.7 |

How much of this is noise: an earlier run of the same backbone configurations on the same split moved test
macro F1 by up to 2 points and validation by up to 1.5; rerunning the basic NN and the custom CNN moved
validation by 1.4 and 1.7 points and test by 0.9 and 0.4; and the four-seed frozen runs below vary by about
1 point on validation and 2 on test (standard deviation). Differences of 2-3 test points between the
stronger models are not rankings.

- **Validation:** fine-tuning the whole ResNet-50 is best (97.9), then the other fine-tuned runs (94-97.5)
  and the custom CNN (95.1). The basic NN (69.5) and the frozen Xception runs (90-91) are clearly behind.
- **Test:** the frozen and stage-1 ResNet-50 (79.6, 80.2), the fine-tuned runs (75.5-78.4) and the custom CNN
  (77.0) all fall in a 5-point band. Fine-tuning raises validation by 2.5-6 points over stage 1; on test it
  helps Xception (74.0 -> 77.1) but not ResNet-50 (80.2 -> 78.4).
- **The largest test errors are the same for every strong model:** `fire` predicted `smokefire`, and `smoke`
  and `smokefire` confused with each other.

### The frozen protocol of the `transfer-learning` branch, for comparison

The branch's own frozen comparison (`src/training/xception_aspect_frozen.py` at `748c653`, plus its nested
cross-validation script) was run unmodified on the same split. It differs from notebook 03 in four ways:
PyTorch `timm` backbones (ResNet-50 `a1_in1k`), both backbones fed [-1, 1] pixels (ResNet-50 expects ImageNet
mean/std), one cached augmented view per run instead of fresh augmentation every epoch, and four seeds.
The branch never scores test; test below was scored afterwards, only for this comparison.
Files: `results/colleague_frozen_protocol/`.

| Protocol | Backbone | Seeds | Val macro F1 | Test macro F1 |
|---|---|---:|---:|---:|
| Notebook 03 | ResNet-50 (Keras) | 1 | 94.1 | 79.6 |
| Notebook 03 | Xception (Keras) | 1 | 90.7 | 75.4 |
| `transfer-learning` branch | ResNet-50 (timm) | 4 | 93.0 +- 1.0 | 77.6 +- 2.3 |
| `transfer-learning` branch | Xception (timm) | 4 | 88.8 +- 1.2 | 71.9 +- 2.1 |
| Branch nested CV, 5 x 4 folds, out-of-fold on train+val | ResNet-50 (timm) | 1 | 94.0 | not scored |
| Branch nested CV, 5 x 4 folds, out-of-fold on train+val | Xception (timm) | 1 | 88.0 | not scored |

Both protocols rank ResNet-50 above Xception, by 3-6 points. Notebook 03 sits 1-2 validation points and
2-3.5 test points above the branch's four-seed mean, 0.9-1.7 seed standard deviations.
Rerunning the branch reproduced its committed four-seed validation means within 0.4 points.

## Known limitations

- **Validation is easier than it looks.** The Kaggle dataset was built by augmenting source photos with
  Keras' `ImageDataGenerator` (stated in the dataset's README), and copies of one photo end up in both train
  and validation. A nearest-neighbour lookup on plain ImageNet features, with no training, scores 97% on
  validation and 74.5% on test. The branch's nested cross-validation splits images, not photos, so it
  shares this problem.
- **Test `smoke` / `smokefire` come from another source:** higher-resolution images, and a few dozen photos
  repeated many times. Errors come in blocks, so the effective test size is far below 800.
- **Labels:** most test `fire` images that every model calls `smokefire` show a large smoke plume next to
  the flames.
- **Confident mistakes:** fine-tuned models put a confidence above 0.9 on most of their test errors.
- One seed per configuration. The full error analysis (`REPORT.md`, notebook 05) is in commit `fc2b1db`.
