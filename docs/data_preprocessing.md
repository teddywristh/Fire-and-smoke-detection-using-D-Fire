# Tài liệu tiền xử lý dữ liệu

Tài liệu này mô tả nguồn dữ liệu, manifest, bước tạo ảnh đã xử lý và các split được dùng trong dự án. Các bước thực tế được triển khai trong [`src/data/data_loader.py`](../src/data/data_loader.py), [`src/data/preprocessing.py`](../src/data/preprocessing.py) và cấu hình tại [`config.py`](../config.py).

## 1. Dataset và nhãn

Dataset chính là Forest Fire C4, gồm các thư mục `train`, `val`, `test`; mỗi split có bốn thư mục lớp:

| Tên lớp | Ý nghĩa | `class_index` |
| --- | --- | ---: |
| `fire` | Có lửa, không thấy khói rõ ràng | 0 |
| `nofire` | Không có lửa hoặc khói | 1 |
| `smoke` | Có khói, không thấy lửa rõ ràng | 2 |
| `smokefire` | Có cả khói và lửa | 3 |

Split gốc có 4.800 ảnh, cân bằng theo lớp:

| Split | Mỗi lớp | Tổng |
| --- | ---: | ---: |
| Train | 800 | 3.200 |
| Validation | 200 | 800 |
| Test | 200 | 800 |

`Forest Fire_Tester` không thuộc các split đã gán nhãn và không được dùng để huấn luyện hoặc đánh giá cuối.

## 2. Cấu hình đường dẫn

`config.py` đọc `DATASET_ROOT`, `TESTER_ROOT` và `DATA_DIR` từ biến môi trường hoặc `local_config.py`. Mặc định dataset nằm ở `../dataset/Forect Fire/Forest Fire_Dataset` so với thư mục project.

Tạo cấu hình riêng cho máy đang chạy:

```powershell
Copy-Item local_config.example.py local_config.py
```

Sau đó sửa `DATASET_ROOT` trong `local_config.py` để trỏ tới thư mục trực tiếp chứa `train/`, `val/` và `test/`. `local_config.py` đã bị `.gitignore` loại trừ; không commit đường dẫn cá nhân.

## 3. Tạo manifest và ảnh processed

Chạy từ thư mục gốc project:

```powershell
python -m src.data.data_loader
```

Pipeline duyệt file `.jpg` theo cấu trúc `split/class`, kiểm tra tên file và metadata ảnh, xác minh ảnh đọc được, tính SHA-256 cho ảnh gốc và ảnh processed, rồi ghi manifest và báo cáo. Ảnh nguồn được xử lý theo các bước:

1. Đọc ảnh bằng Pillow và áp dụng hướng EXIF nếu có.
2. Chuyển ảnh sang RGB.
3. Resize về `config.IMAGE_SIZE = (224, 224)` theo `config.RESIZE_POLICY = "direct_resize"`.
4. Lưu JPEG với quality 95, optimize và progressive encoding.

`direct_resize` ép ảnh về kích thước đích và có thể làm thay đổi tỷ lệ hình học. Hàm `resize_image` cũng hỗ trợ `letterbox`, giữ tỷ lệ và thêm viền đen, nhưng đây không phải policy mặc định của bước tạo ảnh processed. Các pipeline frozen mới đọc ảnh gốc và áp dụng crop/resize runtime riêng; chúng không huấn luyện trên ảnh processed của bước này.

Việc tạo ảnh processed không tạo augmentation. Augmentation chỉ được áp dụng runtime trên train bởi pipeline huấn luyện phù hợp; validation và test không được augmentation.

### Tệp được tạo

- `data/split.csv`: một hàng cho mỗi ảnh, gồm đường dẫn ảnh gốc và processed, split, nhãn, class index, metadata, hash, trạng thái và ghi chú.
- `data/processed/train/`, `data/processed/val/`, `data/processed/test/`: ảnh RGB 224×224 đã resize.
- `data/processed/processing_summary.json` và `processing_report.md`: số lượng, kích thước, duplicate và kết quả kiểm tra.
- `data/processed/samples/`: lưới ảnh ví dụ theo split.

Trong lần xử lý đã ghi nhận tại `processing_summary.json`, pipeline tạo 4.800 ảnh processed, tất cả có trạng thái `ok`, không phát hiện duplicate SHA-256 ở ảnh gốc hoặc processed. Đây là số liệu của bộ dữ liệu hiện có, không phải bảo đảm cho một bản dataset khác.

## 4. Manifest

Các cột chính của `data/split.csv`:

| Cột | Nội dung |
| --- | --- |
| `filepath` | Đường dẫn ảnh gốc |
| `processed_filepath` | Đường dẫn ảnh đã resize |
| `split` | `train`, `val` hoặc `test` |
| `label`, `class_index` | Nhãn chữ và chỉ số lớp |
| `width`, `height`, `mode`, `image_format` | Metadata ảnh nguồn |
| `raw_sha256`, `processed_sha256` | Hash ảnh nguồn và ảnh processed |
| `status`, `note` | Kết quả kiểm tra và cảnh báo |

Các script nên lọc `status == "ok"` trước khi dùng manifest. Dòng `review` cần được kiểm tra trước khi đưa vào run.

## 5. Aspect-aware development split

`data/split_aspect.csv` là manifest riêng cho so sánh frozen backbone. Tạo lại bằng:

```powershell
python -m src.data.create_aspect_split --output data/split_aspect.csv
```

Script chỉ chia lại pool train+validation của manifest gốc, giữ nguyên toàn bộ membership của test và kiểm tra duplicate `raw_sha256` giữa split đầu ra. Mặc định validation chiếm 20% mỗi lớp; seed mặc định là 42. Với dữ liệu hiện tại mỗi lớp vẫn có 800 train và 200 validation. Split này đưa 4 ảnh `smokefire` nhóm wide vào validation và để lại 8 ảnh wide trong train.

Aspect group được xác định từ `width / height`:

| Group | Khoảng tỷ lệ |
| --- | --- |
| `square` | 0.90–1.10 |
| `wide` | 1.60–1.95 |
| `other` | Các tỷ lệ còn lại |

Bốn ảnh `smokefire` wide trong validation chưa đủ để đánh giá độ bền với ảnh wide. Không dùng nhóm nhỏ này làm bằng chứng tổng quát về camera hoặc domain khác.

## 6. Preprocessing phụ thuộc pipeline

Không có một phép biến đổi ảnh duy nhất cho tất cả thí nghiệm:

| Pipeline | Nguồn ảnh | Hình học / kích thước | Chuẩn hóa |
| --- | --- | --- | --- |
| `src.data.data_loader` | Ảnh gốc | Direct resize 224×224, lưu JPEG | RGB; chưa scale tensor |
| Baseline trong `src.training.frozen_transfer` | Ảnh processed | Resize theo backbone; Xception baseline dùng 299×299 | ImageNet mean/std trong transform baseline |
| So sánh frozen ResNet-50/Xception | Ảnh gốc từ `data/split_aspect.csv` | Square crop rồi resize 224×224 | Tensor được scale về `[-1, 1]`; xem lưu ý trong [tài liệu kiến trúc frozen](frozen_transfer_learning_architecture.md) |

Không so sánh metric giữa các hàng trên như thể chỉ thay đổi backbone; nguồn ảnh, crop, kích thước và cách huấn luyện có thể khác.

## 7. Nguyên tắc chống rò rỉ dữ liệu

- Dùng train để học trọng số và augmentation; dùng validation để chọn checkpoint và cấu hình.
- Không dùng test để chọn preprocessing, model, threshold hoặc checkpoint.
- Giữ nguyên test membership khi tạo `split_aspect.csv`; pipeline so sánh frozen hiện tại chỉ load train và validation.
- Các metric test cũ đã được xem trong nhiều thử nghiệm nên chỉ là kết quả lịch sử, không phải ước lượng độc lập cuối cùng.
- Muốn có đánh giá cuối độc lập, cần một test set mới từ nguồn độc lập; chỉ đánh giá sau khi khóa model và preprocessing.

## 8. Tài liệu liên quan

- [Kiến trúc nhánh frozen transfer learning](frozen_transfer_learning_architecture.md)
- [Lịch sử thí nghiệm frozen features và giới hạn đánh giá](frozen_features.md)
- [Notebook so sánh ResNet-50 và Xception](../notebooks/frozen_transfer_learning_pipeline.ipynb)
