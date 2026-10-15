# Phân loại ảnh lửa và khói

Phân loại 4 lớp: `fire`, `nofire`, `smoke`, `smokefire`. Dữ liệu gồm 4.800 ảnh: 3.200 train, 800 validation và 800 test.

## 1. Các notebook

| File | Nội dung |
|---|---|
| `01_basic_nn.ipynb` | Mạng Dense làm baseline |
| `02_custom_cnn.ipynb` | CNN với Residual và Squeeze-and-Excitation |
| `03_frozen_backbone.ipynb` | Xception và ResNet50 đóng băng backbone |
| `04_finetune_backbone.ipynb` | Train head, fine-tune khối cuối hoặc toàn bộ backbone |

Các file nằm trong `notebooks/`, chứa code đọc dữ liệu, xây model, train và đánh giá. Có thể chạy riêng từng notebook. Notebook 04 tự thực hiện các giai đoạn, không cần chạy notebook 03 trước.

## 2. Cài thư viện

Dùng Python 3.11. Mở terminal tại thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter notebook
```

Mở notebook trong Jupyter. Nếu dùng VS Code, chọn kernel Python của `.venv`.

## 3. Đặt dữ liệu

Tải dữ liệu tại [Google Drive của nhóm](https://drive.google.com/drive/folders/1V6pW0pNsZmcDTQw0B2OQO6olNlMldqgB). Giải nén ZIP dữ liệu và đặt `data/processed/` vào repo. Giữ `data/split.csv` đi kèm code.

```text
Fire-and-smoke-detection-using-D-Fire/
├── notebooks/
├── data/
│   ├── split.csv
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
├── results/
├── requirements.txt
└── README.md
```

Trong train/val/test có 4 thư mục lớp. Ví dụ:

```text
data/processed/train/fire/fire_train_1001.jpg
```

Ảnh processed là JPEG RGB 224×224. `split.csv` chỉ chứa đường dẫn, nhãn và cách chia tập, không chứa ảnh. ZIP GitHub không kèm bộ ảnh này. Tránh giải nén thành `data/data/processed/`.

ZIP `data-20261011T054921Z-1-001.zip` đã được kiểm tra: đủ 4.800 ảnh và khớp hash trong manifest hiện tại.

## 4. Chạy notebook

Kiểm tra cấu hình ở bước 1, sau đó chọn **Restart Kernel and Run All Cells**.

### Train lại

`USE_CHECKPOINTS = True` được cố định trong cả 4 notebook, không cần chỉnh. Muốn train lại model nào thì xóa hoặc chuyển file `.keras` của model đó ra khỏi `models/` (tên file ở bảng bên dưới); notebook không thấy checkpoint sẽ tự train.

Riêng notebook 01, giữ `PREPARE_IMAGES = False` để dùng ảnh processed có sẵn.

Notebook sẽ train, chọn checkpoint theo validation loss rồi đánh giá trên test. Train lại sẽ ghi đè model và kết quả cùng tên. Notebook 03–04 cần internet ở lần đầu để tải trọng số ImageNet nếu máy chưa có cache. Nên dùng GPU khi train CNN và transfer learning; CPU sẽ chậm hơn.

### Dùng checkpoint có sẵn

Cần checkpoint đúng tên trong `models/` và `training_history.csv` tương ứng trong `results/`. Thiếu checkpoint thì notebook tự train. Có checkpoint nhưng thiếu history thì bước vẽ biểu đồ sẽ báo lỗi.

| Notebook | Checkpoint trong `models/` |
|---|---|
| 01 | `basic_nn_transfer_preprocessing.keras` |
| 02 | `custom_cnn_transfer_preprocessing.keras` |
| 03 | `frozen_xception_transfer_preprocessing.keras`, `frozen_resnet50_transfer_preprocessing.keras` |
| 04 | `ft_{stage}_{backbone}_transfer_preprocessing.keras` |

Với notebook 04, `stage` là `stage1`, `top`, `all`; `backbone` là `xception`, `resnet50` (6 checkpoint).

Ví dụ, model `frozen_xception_transfer_preprocessing.keras` đi cùng `results/frozen_xception_transfer_preprocessing/training_history.csv`.

**ZIP `models-20261011T054920Z-1-001.zip` thuộc phiên bản khác**, không có các checkpoint trong bảng trên. Không đổi đuôi `.weights.h5` hoặc `.pt` thành `.keras`. Nếu chỉ có ZIP này, train lại hoặc lấy checkpoint đúng phiên bản.

## 5. Chuẩn bị lại ảnh cho Basic NN

Mặc định notebook 01 đọc ảnh processed, resize xuống 64×64 và chuẩn hóa bằng mean/std ImageNet. Lật ngang chỉ hoạt động khi train.

Nếu cần tạo lại ảnh từ dataset gốc, chỉnh ở bước 1:

```python
PREPARE_IMAGES = True
DATASET_ROOT = Path(r"D:\dataset\Forest Fire_Dataset")
```

`DATASET_ROOT` phải chứa train/val/test với tên ảnh khớp `data/split.csv`. Notebook kiểm tra ảnh, chỉnh EXIF, chuyển RGB, resize 224×224 và lưu JPEG chất lượng 95.

Ảnh và manifest mới lưu ở `data/basic_nn_processed/`. Model được train lại, lưu ở `models/basic_nn_prepared.keras` (nếu file này đã có thì notebook dùng lại, muốn train lại thì xóa nó); kết quả ở `results/basic_nn_prepared/`. Bước này chỉ phục vụ Basic NN, không thay đổi dữ liệu của notebook 02–04.

## 6. Kết quả và lỗi thường gặp

Kết quả lưu tại `RESULT_DIR` trong mỗi notebook: metrics, lịch sử train, biểu đồ, confusion matrix, dự đoán và danh sách ảnh sai. Accuracy là tỷ lệ dự đoán đúng; macro F1 là trung bình F1 của 4 lớp.

| Lỗi | Cách xử lý |
|---|---|
| Thiếu ảnh processed | Kiểm tra `data/processed/` và đường dẫn trong `data/split.csv` |
| Không tìm thấy `split.csv` | Mở Jupyter từ gốc repo hoặc chạy trong thư mục `notebooks/` |
| Thiếu `training_history.csv` | Bổ sung history đúng checkpoint hoặc train lại |
| Thiếu thư viện | Cài `requirements.txt` trong đúng môi trường kernel |
| Hết bộ nhớ khi train | Giảm `BATCH_SIZE`, ví dụ từ 32 xuống 16 |

Khi gửi bài, kèm dữ liệu hoặc link tải có quyền truy cập. Muốn người nhận bỏ qua train thì kèm checkpoint và history đúng phiên bản.
