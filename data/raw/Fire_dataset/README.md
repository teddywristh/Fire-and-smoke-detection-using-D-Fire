# 🔥 Forest Fire Image Classification

## Overview
This dataset contains images from forest environments categorized into **4 classes**: **fire, no fire, smoke,** and **smokefire**. It supports research in environmental monitoring, fire detection, and image classification tasks, providing balanced subsets for training, validation, and testing.

---

## 📊 Dataset Details & Structure

### Classes & Images Count
The dataset is organized into three subsets (`train`, `val`, `test`), balanced across all classes:

| Subset | **fire** | **nofire** | **smoke** | **smokefire** | **Total Images** |
|--------|----------|------------|-----------|---------------|------------------|
| **train** | 800      | 800        | 800       | 800           | 3,200            |
| **val**   | 200      | 200        | 200       | 200           | 800              |
| **test**  | 200      | 200        | 200       | 200           | 800              |
| **Forest Fire Tester** | - | - | - | - | 23 |

**Total Images**: 4,823  
**Format**: JPEG  
**Dimensions**: 250x250 pixels  

---

## 🔄 Data Augmentation & Preprocessing

### Image Augmentation
To enhance data diversity and robustness, images were augmented using Keras's `ImageDataGenerator`. Below is the code and parameters used for augmentation:

```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.2, 1.2]
)
```


The augmentations include:
- **Rotation**: Up to 10 degrees.
- **Width & Height Shift**: Up to 10% of the image size.
- **Shearing & Zooming**: 10% variation.
- **Horizontal Flip**: Randomly flips images.
- **Brightness Adjustment**: Brightness variations from 0.2 to 1.2.

### Image Cropping & Standardization
- All images were resized to a uniform **250x250 pixels**.
- The dataset maintains consistent file naming, such as `<class>_<subset>_<serial_number>.jpg` (e.g., `fire_train_001.jpg`).

---

## 📂 Class Details

### 1. Train, Val, Test Folders
These folders contain images sorted into 4 classes representing different forest conditions.

| **Path**              | **Class**   | **Description**                                  |
|-----------------------|-------------|--------------------------------------------------|
| **/train/fire**       | Fire        | Images showing visible fire/flames.              |
| **/train/nofire**     | No Fire     | Forest scenes with no fire present.              |
| **/train/smoke**      | Smoke       | Images depicting smoke but no visible flames.    |
| **/train/smokefire**  | SmokeFire   | Images with both smoke and visible flames.       |

The structure and contents are mirrored across the `val` and `test` folders for validation and evaluation.

### 2. Forest Fire Tester Folder
- **Path**: `/Forest Fire_Tester`
- **Description**: Contains a small set of **23 images** for quick manual testing and evaluation, named sequentially (`1.jpg`, `2.jpg`, ...).

---

## 📝 How to Use the Dataset

- **Training & Validation**: Use the `train` and `val` folders for building and fine-tuning machine learning models.
- **Testing**: Use the `test` folder for final model evaluation.
- **Manual Testing**: The `Forest Fire_Tester` folder contains a small set of images for manual evaluation.

Check out a live working sample here: [Forest Fire Live Sample](https://osnaren.github.io/ForestFire)

---

## 📝 Citation

If you use this dataset, please cite it as follows:

**APA**  
Obuli Sai Naren. (2022). *Forest Fire Image Classification Dataset* [Data set]. Kaggle. [https://doi.org/10.34740/KAGGLE/DSV/3135325](https://doi.org/10.34740/KAGGLE/DSV/3135325)

---

Feel free to download, explore, and contribute! 📊💻