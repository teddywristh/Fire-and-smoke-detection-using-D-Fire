# Phân loại ảnh cháy rừng và khói

Dự án so sánh Basic Neural Network, Custom CNN và Transfer Learning
(frozen backbone, fine-tuning) trên Forest Fire Image Classification Dataset.
Hiện đã có pipeline dữ liệu và **một Basic NN hoàn chỉnh**. Các mô hình còn lại
chưa được triển khai; thư mục notebooks/demo hiện chưa có notebook chạy được.

## Dữ liệu

Nguồn: [Forest Fire Image Classification Dataset](https://www.kaggle.com/datasets/obulisainaren/forest-fire-c4).
Tên repo có D-Fire nhưng pipeline hiện dùng bộ dữ liệu forest-fire-c4 với bốn lớp:
`fire`, `nofire`, `smoke`, `smokefire`.
Giữ split gốc: train 3.200, validation 800, test 800 ảnh.

Mặc định dataset nằm tại `../dataset/Forect Fire/Forest Fire_Dataset`,
thư mục này chứa trực tiếp `train/`, `val/`, `test/`.
Nếu dùng vị trí khác, tạo `local_config.py` ở thư mục gốc:

```python
from pathlib import Path

DATASET_ROOT = Path(r"D:/dataset/Forest Fire_Dataset")
# Chỉ cần khi dùng ảnh demo:
TESTER_ROOT = Path(r"D:/dataset/Forest Fire_Tester")
```

File này được Git bỏ qua. Có thể dùng biến môi trường
`FOREST_FIRE_DATASET_ROOT`, `FOREST_FIRE_TESTER_ROOT`, `FOREST_FIRE_DATA_DIR`
để ghi đè đường dẫn. Không cần sửa `config.py` khi đổi máy.

## Chạy Basic NN

Mở terminal ở thư mục gốc dự án, dùng cùng môi trường Python cho các lệnh:

```powershell
python -m pip install -r requirements.txt
python -m src.data_loader
python -m src.basic_nn
```

Chỉ cần chuẩn bị dữ liệu khi chưa có `data/split.csv` và ảnh processed,
hoặc khi đổi dataset. Lệnh Basic NN tự train, chọn checkpoint theo validation
loss, đánh giá test và xuất báo cáo. Train lại sẽ thay model và kết quả hiện tại.

Chỉ đánh giá lại model đã lưu:

```powershell
python -m src.basic_nn --evaluate-only
```

Chỉ cập nhật Markdown kết quả từ CSV/JSON đã lưu; tài liệu phương pháp được giữ riêng:

```powershell
python -m src.basic_nn --report-only
```

## Cấu hình và kết quả

Ảnh processed RGB 224×224 được resize về 64×64 khi đưa vào Basic NN,
scale pixel về [0, 1]. Kiến trúc: Flatten → Dense(256, ReLU) → Dropout(0.5)
→ Dense(4, Softmax). Adam learning rate 0.0005, batch size 32,
tối đa 30 epochs, EarlyStopping patience 5, seed 42.
Augmentation nhẹ chỉ áp dụng cho train: lật ngang, brightness, contrast.

Checkpoint được giữ sau tuning bằng validation đạt test accuracy **59,75%**, Macro F1 **0,6006**,
322 lỗi trên 800 ảnh. Dùng thêm ReduceLROnPlateau (factor 0.5, patience 2,
min_lr 0.00001). Đây là kết quả lần chạy đã lưu; chạy lại có thể khác.
Cấu hình gốc từng được chọn theo test lịch sử; đợt tuning mới chọn bằng validation
loss rồi mới đánh giá test. Cần holdout mới để đánh giá độc lập sau lựa chọn lịch sử.
So sánh chi tiết nằm trong báo cáo kết quả.

## Tệp chính

| Tệp | Vai trò |
|---|---|
| `config.py` | Đường dẫn, split, nhãn, chính sách xử lý dữ liệu |
| `src/data_loader.py` | Chuẩn bị ảnh, manifest, kiểm tra SHA256 |
| `src/preprocessing.py` | RGB, EXIF, resize, lưu JPEG |
| `src/basic_nn.py` | Toàn bộ Basic NN: train, đánh giá, biểu đồ và báo cáo |
| `models/basic_nn_best.keras` | Checkpoint Basic NN duy nhất |
| `results/basic_nn/` | CSV, JSON, biểu đồ và báo cáo |

Model và ảnh processed không được Git theo dõi theo `.gitignore`;
khi clone sang máy khác cần chuẩn bị dữ liệu và train, hoặc sao chép checkpoint.

## Tài liệu

- [Phương pháp Basic NN](results/basic_nn/basic_nn_method.md): dữ liệu, kiến trúc, quy trình và cách chạy.
- [Kết quả Basic NN](results/basic_nn/basic_nn_results.md): huấn luyện, metrics, confusion matrix, phân tích lỗi và so sánh lịch sử.
- [Xử lý dữ liệu](DATA_PROCESSING.md): cấu hình và quy trình pipeline đang dùng.
- [Log xử lý dữ liệu](data/processed/processing_report.md): tự sinh bởi data loader.

Các file `augmentation.py`, `evaluation.py`, `visualization.py`, `gradcam.py`
hiện là phần chuẩn bị cho các phương pháp tiếp theo. Grad-CAM dự kiến dùng
cho CNN/Transfer Learning; chưa có kết quả so sánh các mô hình này.
