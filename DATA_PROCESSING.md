# Tong hop xu ly dataset

Tai lieu nay tong hop tinh trang dataset hien tai va ke hoach xu ly data cho project phan loai anh chay rung va khoi. Noi dung nay dung de chuan bi truoc khi viet code data pipeline.

## 1. Vi tri dataset

Dataset hien dang nam cung cap voi thu muc project:

```text
../dataset/
```

Ben trong dataset:

```text
../dataset/
|-- README.md
`-- Forect Fire/
    |-- Forest Fire_Dataset/
    |   |-- train/
    |   |   |-- fire/
    |   |   |-- nofire/
    |   |   |-- smoke/
    |   |   `-- smokefire/
    |   |-- val/
    |   |   |-- fire/
    |   |   |-- nofire/
    |   |   |-- smoke/
    |   |   `-- smokefire/
    |   `-- test/
    |       |-- fire/
    |       |-- nofire/
    |       |-- smoke/
    |       `-- smokefire/
    `-- Forest Fire_Tester/
```

Luu y: ten thu muc `Forect Fire` co ve bi sai chinh ta, nhung tam thoi nen giu nguyen de tranh sai duong dan.

## 2. Cau truc split va so luong anh

Dataset chinh nam trong:

```text
../dataset/Forect Fire/Forest Fire_Dataset
```

Bang so luong:

| Split | fire | nofire | smoke | smokefire | Tong |
|---|---:|---:|---:|---:|---:|
| train | 800 | 800 | 800 | 800 | 3200 |
| val | 200 | 200 | 200 | 200 | 800 |
| test | 200 | 200 | 200 | 200 | 800 |

Tong dataset chinh: 4800 anh.

Moi lop co tong cong 1200 anh. Dataset da can bang nen ban dau chua can dung class weight.

## 3. Y nghia cac lop

| Lop | Y nghia |
|---|---|
| fire | Co lua ro rang, khong tap trung vao khoi |
| nofire | Khong co lua va khong co khoi |
| smoke | Co khoi, khong thay lua ro |
| smokefire | Co ca khoi va lua |

Label nen duoc giu thong nhat:

```text
fire
nofire
smoke
smokefire
```

Mapping de xuat khi code:

| Class | Index |
|---|---:|
| fire | 0 |
| nofire | 1 |
| smoke | 2 |
| smokefire | 3 |

## 4. Ket qua kiem tra dataset

Nhung diem tot:

- Dataset `train/val/test` can bang theo lop.
- Ten file nhat quan theo mau `<class>_<split>_<number>.jpg`.
- Tat ca file trong `Forest Fire_Dataset` doc duoc.
- Khong phat hien file trung lap chinh xac trong `train/val/test`.
- Tat ca anh trong dataset chinh co header JPEG hop le.

Nhung diem can chu y:

- README cua dataset noi anh co kich thuoc `250x250`, nhung thuc te khong phai tat ca deu `250x250`.
- Lop `smoke` trong ca `train/val/test` khong phai `250x250`.
- `test/smokefire` cung khong phai `250x250`.
- Mot so anh co vet keo gian o mep anh, co the la artifact do resize/crop truoc do.
- Thu muc `Forest Fire_Tester` khong co nhan chuan, chi nen dung de demo thu cong.
- File `../dataset/Forect Fire/Forest Fire_Tester/3.jpg` thuc chat la WEBP bi dat duoi `.jpg`, nen khong nen dung truc tiep.

## 5. Nguyen tac xu ly data

1. Giu nguyen split co san.
   - `train`: dung de train model.
   - `val`: dung de chon model, early stopping, tuning.
   - `test`: chi dung de danh gia cuoi cung.

2. Khong dua `Forest Fire_Tester` vao training hoac final evaluation.
   - Thu muc nay chi nen dung sau khi da co model de demo thu cong.
   - File `3.jpg` trong thu muc tester can bo qua hoac convert dung dinh dang truoc khi demo.

3. Tao manifest chung tai `data/split.csv`.
   - File nay nen luu duong dan anh, split, label, class index, width, height va ghi chu kiem tra.
   - Manifest giup notebook va script dung chung mot nguon split.

4. Chuan hoa tat ca anh ve cung kich thuoc truoc khi dua vao model.
   - Kich thuoc de xuat: `224x224` neu dung transfer learning.
   - Tat ca model nen dung cung mot kich thuoc input de so sanh cong bang.
   - Anh nen duoc chuyen ve RGB.

5. Can than voi anh khong vuong.
   - Neu resize truc tiep, anh co the bi meo.
   - De giam meo hinh, nen uu tien resize giu ti le roi padding, hoac ap dung mot chinh sach crop/pad nhat quan.

6. Augmentation chi ap dung cho train.
   - Co the dung horizontal flip, rotation nhe, zoom nhe, shift nhe, brightness/contrast nhe.
   - Khong augmentation cho val va test.
   - Khong nen bien doi qua manh vi co the lam mat lua/khoi hoac tao tin hieu gia.

7. Normalization phai nhat quan.
   - Model tu xay co the dung scale pixel ve `[0, 1]`.
   - Transfer learning nen dung preprocessing dung voi backbone duoc chon.

8. Danh gia phai tap trung vao loi nguy hiem.
   - `fire` bi du doan thanh `nofire`.
   - `smokefire` bi du doan thanh `nofire`.
   - `fire` nham voi `smokefire`.
   - `smoke` nham voi `smokefire`.

## 6. Pipeline xu ly data de xuat

Thu tu nen lam:

1. Doc cau truc thu muc dataset.
2. Kiem tra split va ten lop.
3. Kiem tra file anh co doc duoc khong.
4. Lay thong tin width, height, dinh dang, dung luong.
5. Tao `data/split.csv`.
6. Xay loader cho `train`, `val`, `test`.
7. Them preprocessing resize, RGB, normalization.
8. Them augmentation rieng cho train.
9. Kiem tra batch mau truoc khi train.
10. Luu lai thong ke va hinh visualize de bao cao.

## 7. Structure da chuan bi trong project

```text
Fire-and-smoke-detection-using-D-Fire/
|-- DATA_PROCESSING.md
|-- config.py
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- split.csv
|-- demo/
|-- models/
|-- notebooks/
|-- results/
|   |-- basic_nn/
|   |-- custom_cnn/
|   |-- transfer_frozen/
|   `-- transfer_finetuned/
`-- src/
    |-- __init__.py
    |-- augmentation.py
    |-- data_loader.py
    |-- evaluation.py
    |-- gradcam.py
    |-- preprocessing.py
    `-- visualization.py
```

## 8. Checklist truoc khi code

- [x] Xac nhan duong dan dataset dung la `../dataset/Forect Fire/Forest Fire_Dataset`.
- [x] Quyet dinh kich thuoc input chung: `224x224`.
- [x] Quyet dinh cach resize anh khong vuong: `direct_resize`.
- [x] Tao manifest day du vao `data/split.csv`.
- [x] Kiem tra processed images cua train/val/test sau preprocessing.
- [x] Visualize mau anh moi lop sau resize bang sample grids.
- [ ] Bat dau train Basic Neural Network.
- [ ] Train Custom CNN.
- [ ] Train Transfer Learning Frozen.
- [ ] Fine-tune tu model Frozen tot nhat.
- [ ] So sanh bang metrics va confusion matrix.
- [ ] Chay Grad-CAM cho model cuoi.

## 9. Ket qua xu ly da tao

Da chay pipeline xu ly data bang lenh:

```bash
python -m src.data_loader
```

Cac file/thuc muc chinh da sinh ra:

```text
data/split.csv
data/processed/train/
data/processed/val/
data/processed/test/
data/processed/processing_report.md
data/processed/processing_summary.json
data/processed/samples/train_grid.jpg
data/processed/samples/val_grid.jpg
data/processed/samples/test_grid.jpg
```

Ket qua kiem tra:

| Hang muc | Ket qua |
|---|---:|
| Anh raw duoc dua vao manifest | 4800 |
| Anh processed da tao | 4800 |
| Kich thuoc processed | 224x224 |
| Anh processed doc loi | 0 |
| Dong manifest status `ok` | 4800 |
| Duplicate raw theo SHA256 | 0 |
| Duplicate processed theo SHA256 | 0 |

So luong processed theo split/lop:

| Split | fire | nofire | smoke | smokefire | Tong |
|---|---:|---:|---:|---:|---:|
| train | 800 | 800 | 800 | 800 | 3200 |
| val | 200 | 200 | 200 | 200 | 800 |
| test | 200 | 200 | 200 | 200 | 800 |

## 10. Ly do chon `direct_resize`

Raw dataset co kich thuoc anh khac nhau va kich thuoc nay co tuong quan voi lop. Neu dung padding de giu ti le, mot so lop se co vien padding dac trung hon cac lop khac. Khi do model co the hoc shortcut tu vien anh thay vi hoc lua/khoi.

Vi vay pipeline da chon `direct_resize` ve `224x224`:

- Tat ca model nhan input cung kich thuoc.
- Khong tao them vien padding co nguy co thanh tin hieu gia.
- Giu pipeline don gian, de giai thich va de tai lap.
- Split goc van duoc giu nguyen, khong tron train/val/test.

## 11. Nguyen tac chong data leak da ap dung

- Khong random split lai dataset.
- Khong dua anh `val` hoac `test` vao `train`.
- Khong dua `Forest Fire_Tester` vao train/val/test.
- Khong tao augmentation ra dia cho `val` va `test`.
- Khong tinh preprocessing dua tren thong tin cua `val` hoac `test`.
- Kiem tra duplicate SHA256 tren raw va processed: deu bang 0.
- Manifest luu ro `split`, `label`, `class_index`, duong dan raw va duong dan processed.

## 12. Cach dung cho buoc tiep theo

Khi train model, nen doc tu `data/split.csv` va dung cot:

- `processed_filepath`: duong dan anh da xu ly.
- `split`: `train`, `val`, hoac `test`.
- `label`: ten lop.
- `class_index`: nhan so.

Augmentation chi ap dung runtime cho cac dong co `split == "train"`. Validation va test chi resize/normalize theo dung preprocessing cua model, khong augmentation.

## 13. Doi duong dan dataset khi lam nhom

Khong nen sua truc tiep `config.py` moi khi doi may, vi file nay se duoc push len Git. Cach khuyen dung:

1. Copy file mau:

```bash
copy local_config.example.py local_config.py
```

2. Sua `DATASET_ROOT` trong `local_config.py` cho dung may cua ban:

```python
from pathlib import Path

DATASET_ROOT = Path(r"D:/your/path/to/dataset/Forect Fire/Forest Fire_Dataset")
TESTER_ROOT = Path(r"D:/your/path/to/dataset/Forect Fire/Forest Fire_Tester")
```

3. Chay lai pipeline:

```bash
python -m src.data_loader
```

`local_config.py` da nam trong `.gitignore`, nen moi thanh vien co the dat duong dan rieng ma khong lam thay doi code cua nhom.

Neu muon dung bien moi truong thay vi `local_config.py`, co the set:

```text
FOREST_FIRE_DATASET_ROOT
FOREST_FIRE_TESTER_ROOT
FOREST_FIRE_DATA_DIR
```
