# Xử lý dữ liệu

Tài liệu mô tả pipeline đã triển khai trong `src/data_loader.py` và
`src/preprocessing.py`. Kết quả Basic NN và phân tích thử nghiệm được tổng hợp
tại [báo cáo Basic NN](results/basic_nn/basic_nn_results.md).

## Nguồn và nhãn

Dataset mặc định: `../dataset/Forect Fire/Forest Fire_Dataset`.
Giữ nguyên tên thư mục `Forect Fire` để khớp dữ liệu hiện có.

| Nhãn | Index | Ý nghĩa |
|---|---:|---|
| fire | 0 | Có lửa, không thấy khói rõ |
| nofire | 1 | Không có lửa hoặc khói |
| smoke | 2 | Có khói, không thấy lửa rõ |
| smokefire | 3 | Có cả khói và lửa |

| Split | fire | nofire | smoke | smokefire | Tổng |
|---|---:|---:|---:|---:|---:|
| train | 800 | 800 | 800 | 800 | 3200 |
| val | 200 | 200 | 200 | 200 | 800 |
| test | 200 | 200 | 200 | 200 | 800 |

Các số trên lấy từ `data/processed/processing_summary.json` đã lưu.
`Forest Fire_Tester` không có nhãn chuẩn, chỉ dùng demo; không đưa vào train/test.

## Pipeline hiện tại

1. Duyệt các thư mục split/lớp, kiểm tra tên file và khả năng đọc ảnh.
2. Ghi metadata, đường dẫn và SHA256 vào `data/split.csv`.
3. Áp dụng EXIF orientation, chuyển RGB, direct resize về 224×224.
4. Lưu JPEG chất lượng 95, giữ nguyên cấu trúc split/lớp.
5. Ghi báo cáo, thống kê JSON và ảnh mẫu cho từng split.

Direct resize tránh thêm viền padding có thể trở thành shortcut vì kích thước
ảnh gốc có tương quan với lớp, nhưng có thể làm méo ảnh không vuông.
224×224 là kích thước lưu chung; mỗi mô hình có thể resize runtime theo kiến trúc.
Basic NN hiện dùng 64×64, chuẩn hóa pixel [0, 1].

## Kết quả kiểm tra đã lưu

- 4.800 ảnh trong manifest, toàn bộ `status=ok`.
- Không phát hiện bản trùng chính xác theo SHA256 ở raw hoặc processed.
- Ảnh gốc có nhiều kích thước; 3.336 ảnh là 250×250.
- SHA256 không phát hiện được mọi ảnh gần giống hoặc chụp cùng cảnh.

Log chi tiết: [processing_report.md](data/processed/processing_report.md).

## Nguyên tắc khi train

Giữ nguyên split gốc. Train dùng cập nhật trọng số; validation dùng checkpoint
và EarlyStopping. Augmentation chỉ thực hiện runtime cho train; không lưu ảnh
augmentation vào val/test. Test không shuffle để dự đoán khớp thứ tự manifest.

Model đọc `processed_filepath`, `split`, `label`, `class_index`, `status`;
chỉ dùng dòng `status=ok`. Mọi mô hình cần giữ cùng split và mapping nhãn,
nhưng preprocessing có thể khác theo kiến trúc.

Basic NN đã được chọn hồi cứu theo test của các thử nghiệm đã chạy.
Vì vậy các số test hiện tại không phải đánh giá độc lập sau chọn cấu hình.

## Cấu hình đường dẫn và chạy

Tạo `local_config.py` ở thư mục gốc nếu dataset khác vị trí mặc định:

```python
from pathlib import Path

DATASET_ROOT = Path(r"D:/dataset/Forest Fire_Dataset")
TESTER_ROOT = Path(r"D:/dataset/Forest Fire_Tester")
```

`local_config.py` không được Git theo dõi. Biến môi trường
`FOREST_FIRE_DATASET_ROOT`, `FOREST_FIRE_TESTER_ROOT`, `FOREST_FIRE_DATA_DIR`
được ưu tiên hơn file này. Không cần sửa cấu hình dùng chung khi đổi máy.

```powershell
python -m src.data_loader
```

Đầu ra: manifest `data/split.csv`, ảnh trong `data/processed/{train,val,test}/`,
`processing_summary.json`, `processing_report.md`, và sample grids.
Sau đó chạy `python -m src.basic_nn` theo hướng dẫn trong README.
