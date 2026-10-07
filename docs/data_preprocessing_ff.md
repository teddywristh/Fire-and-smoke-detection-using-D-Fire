# Data preprocessing cho Frozen Features

Tai lieu nay mo ta nguon du lieu, manifest va cac cach tien xu ly anh duoc dung trong nhanh PyTorch Frozen Transfer Learning. Du lieu goc va split Kaggle duoc giu rieng; cac run frozen co the dung preprocessing khac nhau theo yeu cau cua backbone. Bao cao tong hop model va metric nam trong [`docs/frozen_features.md`](docs/frozen_features.md).

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
|-- data_preprocessing_ff.md
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
    |-- data/
    |   |-- data_loader.py
    |   |-- download_dataset.py
    |   |-- create_aspect_split.py
    |   `-- preprocessing.py
    |-- training/
    |   |-- frozen_transfer.py
    |   |-- frozen_multilabel.py
    |   |-- fine_tune_transfer.py
    |   |-- xception_aspect_frozen.py
    |   `-- calibrate_frozen.py
    `-- analysis/
        |-- augmentation.py
        |-- evaluation.py
        |-- gradcam.py
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
python -m src.data.data_loader
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

## 10. Preprocessing theo tung frozen-features pipeline

Khong co mot resize policy chung cho moi run frozen. Cac pipeline da dung la:

| Pipeline | Input | Hinh hoc | Normalize | Ghi chu |
|---|---:|---|---|---|
| ResNet-50/EfficientNet/MobileNet baseline | 224x224 | Direct resize | ImageNet mean/std `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` | Doc anh processed tu `data/split.csv` |
| Xception baseline (`src.training.frozen_transfer`) | 299x299 | Resize truc tiep | ImageNet mean/std chung | Run lich su; normalization khong theo cfg Xception |
| Xception aspect letterbox (`src.training.xception_aspect_frozen`) | 64/224/256/299 | Giu aspect, padding den | Xception mean/std `(0.5, 0.5, 0.5)` | Vien padding co the de lo aspect ratio |
| Xception aspect square (`src.training.xception_aspect_frozen`) | 224x224 | Train random square crop; val center square crop | Xception mean/std `(0.5, 0.5, 0.5)`, pixel vao `[-1,1]` | Run hien tai; khong tao vien den tu aspect padding |

Square crop loai bo pattern vien den ma letterbox tao ra, nhung co the cat mat noi dung o canh anh. No khong xoa cac domain cues khac nhu camera, anh sang, mau sac hay boi canh. Vi vay so sanh square voi letterbox can dung validation dai dien, khong chon policy dua tren test cu.

## 11. Split va nguyen tac chong data leakage

- `data/split.csv` giu nguyen split Kaggle cho cac baseline va cac bao cao lich su.
- `data/split_aspect.csv` tao validation moi bang cach chia lai pool train+val; test membership duoc giu nguyen va khong duoc lay mau de tao split.
- Khong dua anh validation hay test vao training.
- Khong dua `Forest Fire_Tester` vao train/val/test.
- Khong tao augmentation ra dia cho `val` va `test`.
- Augmentation runtime chi ap dung train; validation dung deterministic center crop/resize theo policy cua model.
- Kiem tra duplicate SHA256 tren raw va processed: deu bang 0.
- Manifest luu ro `split`, `label`, `class_index`, duong dan raw va duong dan processed.

Luu y: test goc da duoc xem qua de so sanh nhieu cau hinh cu, nen cac metric test do chi la ket qua lich su va khong con la danh gia doc lap. Run square-crop moi chi dung train/validation de cache feature va chon checkpoint; chua nap anh/features test.

## 12. Cach dung cho buoc tiep theo

Baseline frozen va cac run cu doc `data/split.csv`; run square-crop moi doc `data/split_aspect.csv`. Cac cot chinh:

- `processed_filepath`: duong dan anh da xu ly.
- `split`: `train`, `val`, hoac `test`.
- `label`: ten lop.
- `class_index`: nhan so.

Augmentation chi ap dung runtime cho cac dong co `split == "train"`. Validation dung preprocessing deterministic. Chi danh gia test sau khi model/preprocessing duoc khoa; vi test Kaggle da bi xem trong cac run cu, can mot test set ngoai doc lap de co uoc luong cuoi cung khong bi anh huong boi qua trinh chon model.

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
python -m src.data.data_loader
```

`local_config.py` da nam trong `.gitignore`, nen moi thanh vien co the dat duong dan rieng ma khong lam thay doi code cua nhom.

Neu muon dung bien moi truong thay vi `local_config.py`, co the set:

```text
FOREST_FIRE_DATASET_ROOT
FOREST_FIRE_TESTER_ROOT
FOREST_FIRE_DATA_DIR
```

## 14. Split validation theo lop va aspect cho thu nghiem Xception

De validation co mot so anh `smokefire` wide, co the tao manifest rieng tu pool train/val cu. Anh trong split test duoc giu nguyen va khong tham gia qua trinh tao validation:

```powershell
python -m src.data.create_aspect_split --output data/split_aspect.csv
```

Voi du lieu hien tai, validation moi co 4 anh `smokefire` wide va train con 8 anh. So luong nay chi cho phep kiem tra pipeline, chua du de uoc luong hieu nang wide mot cach on dinh; can thu thap them anh wide co nhan tu nguon khac, khong lay tu Kaggle test.

Chay frozen Xception tren split phat trien moi, square-crop ca train/validation, va lay mau co trong so theo cap (class, aspect group):

```powershell
python -m src.training.xception_aspect_frozen --split-csv data/split_aspect.csv --crop-mode square --aspect-balance --image-size 224 --epochs 30 --batch-size 32 --cpu-threads 4
```

Square crop bo vien den va dua moi input ve cung hinh hoc vuong; validation dung center crop, train dung random square crop. Crop co the bo mat thong tin gan canh anh va khong xoa moi tuong quan noi dung voi camera/domain. Ket qua va checkpoint moi duoc ghi rieng trong `results/transfer_frozen/` va `models/transfer_frozen/`. Luong nay chi trich xuat feature train/validation; khong nap anh/features test hoac danh gia test. Chon preprocessing/model bang validation, sau khi khoa quyet dinh moi danh gia test mot lan.
