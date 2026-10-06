# Hướng Dẫn Phát Triển Mô Hình Advanced Complex CNN

**Nhánh:** `custom-complex-cnn`

## 1. Tổng quan

Mục tiêu của phần này là xây dựng một mô hình CNN nâng cao để phân loại ảnh cháy rừng và khói. Mô hình được train **from scratch**, tức là không sử dụng các trọng số đã được train trước.

Bài toán gồm 4 lớp:

* `0: fire`: Có lửa nhưng không thấy rõ khói.
* `1: nofire`: Cảnh rừng bình thường, không có lửa và khói.
* `2: smoke`: Có khói nhưng không nhìn thấy lửa.
* `3: smokefire`: Có cả khói và lửa.

Mô hình này sẽ được sử dụng để so sánh với mô hình Baseline và các mô hình Transfer Learning như EfficientNet và MobileNet.

---

## 2. Dữ liệu đầu vào

Dữ liệu đã được xử lý ở các bước trước nên khi đưa vào model có các thông số:

* Kích thước ảnh: `(224, 224, 3)`.
* Ảnh có 3 kênh màu RGB.
* Giá trị pixel được chuẩn hóa về khoảng `[0.0, 1.0]`.
* Label được chuyển sang dạng One-hot với 4 lớp.

Dữ liệu được chia thành:

| Dataset    |  Số lượng | Ghi chú     |
| ---------- | --------: | ----------- |
| Train      | 3.200 ảnh | 800 ảnh/lớp |
| Validation |   800 ảnh | 200 ảnh/lớp |
| Test       |   800 ảnh | 200 ảnh/lớp |

Tập train có sử dụng augmentation trong quá trình huấn luyện. Tập validation và test không sử dụng augmentation.

Batch size có thể chọn `32` hoặc `64`.

---

# 3. Thiết kế mô hình Advanced Complex CNN

## 3.1. Ý tưởng

Nếu chỉ sử dụng nhiều lớp `Conv2D` nối tiếp nhau thì khi mạng trở nên sâu có thể gặp vấn đề về gradient và việc train sẽ khó hơn.

Ngoài ra, ảnh cháy rừng có nhiều đặc trưng khác nhau. Lửa thường có màu đỏ, vàng và các cạnh khá rõ, trong khi khói thường có màu xám hoặc trắng, hình dạng mờ và trải rộng. Phần nền như cây cối hoặc đất cũng có thể làm model bị nhầm.

Vì vậy, model sẽ kết hợp các thành phần sau:

### Stem Block

Stem Block là phần đầu tiên của mạng. Nó sử dụng convolution với `stride=2` để giảm kích thước ảnh nhưng vẫn giữ lại các đặc trưng quan trọng.

### Residual Connection

Residual Connection giúp model truyền thông tin từ đầu vào sang đầu ra thông qua một shortcut:

```text
y = F(x) + x
```

Điều này giúp gradient truyền qua mạng dễ hơn khi model có nhiều layer.

### Squeeze-and-Excitation (SE)

SE Block giúp model chú ý hơn đến những channel quan trọng. Ví dụ, một số channel có thể chứa thông tin về lửa hoặc khói, trong khi các channel khác có thể chủ yếu chứa thông tin nền.

### Global Average Pooling

Thay vì dùng `Flatten` với số lượng tham số lớn, model sử dụng `GlobalAveragePooling2D`.

GAP lấy giá trị trung bình của từng feature map, giúp giảm số lượng tham số và hạn chế overfitting.

---

## 3.2. Các block chính

### 3.2.1. Squeeze-and-Excitation Block

Đầu vào của SE Block có dạng:

```text
(H, W, C)
```

Đầu tiên là bước **Squeeze**:

```python
GlobalAveragePooling2D()
```

Sau bước này feature map được đưa về dạng:

```text
(1, 1, C)
```

Tiếp theo là bước **Excitation**:

```python
Dense(C // ratio, activation='relu')
Dense(C, activation='sigmoid')
```

Kết quả là các trọng số cho từng channel.

Cuối cùng, các trọng số này được nhân với feature map ban đầu. Những channel quan trọng sẽ được giữ lại nhiều hơn.

---

### 3.2.2. Residual-SE Block

Block này kết hợp Residual Block và SE Block.

#### Main Path

Main path gồm:

```text
Conv2D(filters, 3x3, stride=stride)
        ↓
BatchNormalization
        ↓
Activation (Swish hoặc ReLU)
        ↓
Conv2D(filters, 3x3, stride=1)
        ↓
BatchNormalization
        ↓
SE Block
```

Các lớp convolution sử dụng `padding='same'` và `use_bias=False`.

#### Shortcut Path

Nếu `stride == 1` và số channel không thay đổi thì có thể giữ nguyên input.

Nếu `stride > 1` hoặc số channel thay đổi thì sử dụng:

```text
Conv2D(filters, 1x1, strides=stride)
        ↓
BatchNormalization
```

Sau đó cộng hai nhánh:

```python
Add()([main_path, shortcut_path])
```

Cuối cùng sử dụng activation là `Swish` hoặc `ReLU`.

---

# 4. Kiến trúc tổng thể

Model có thể được xây dựng theo các phần sau.

## Input

```text
(224, 224, 3)
```

## Stem Layer

```text
Conv2D(32, 3x3, stride=2)
→ BatchNormalization
→ Swish
```

Output:

```text
(112, 112, 32)
```

## Stage 1

Sử dụng 1 hoặc 2 Res-SE Block:

```text
filters = 64
stride = 1
```

Output:

```text
(112, 112, 64)
```

## Stage 2

Sử dụng 2 Res-SE Block:

```text
filters = 128
```

Block đầu tiên sử dụng `stride=2`.

Output:

```text
(56, 56, 128)
```

## Stage 3

Sử dụng 2 Res-SE Block:

```text
filters = 256
```

Block đầu tiên sử dụng `stride=2`.

Output:

```text
(28, 28, 256)
```

## Classification Head

Phần cuối của model:

```text
GlobalAveragePooling2D()
        ↓
Dropout(0.3 - 0.5)
        ↓
Dense(128, activation='swish')
        ↓
Dense(4, activation='softmax')
```

Lớp cuối có 4 neuron tương ứng với 4 lớp:

```text
fire
nofire
smoke
smokefire
```

---

# 5. Huấn luyện model

Phần train được thực hiện trong file `train.py`.

## 5.1. Loss Function

Sử dụng:

```python
CategoricalCrossentropy(label_smoothing=0.05)
```

Label smoothing giúp model không quá tự tin khi gặp những ảnh khó phân biệt, ví dụ giữa `smoke` và `smokefire`.

---

## 5.2. Optimizer

Có thể sử dụng `AdamW`:

```python
AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4
)
```

Hoặc sử dụng:

```python
Adam(learning_rate=1e-3)
```

---

## 5.3. Callbacks

### ModelCheckpoint

Lưu lại model tốt nhất dựa trên `val_loss`:

```text
models/custom_cnn_best.keras
```

### EarlyStopping

Nếu `val_loss` không cải thiện trong khoảng 12–15 epochs thì dừng train.

Sử dụng:

```python
restore_best_weights=True
```

để lấy lại trọng số tốt nhất.

### ReduceLROnPlateau

Nếu `val_loss` không cải thiện sau 4 epochs thì giảm learning rate xuống 50%.

Các thông số:

```text
factor = 0.5
patience = 4
min_lr = 1e-6
```

### CSVLogger

Lưu các thông tin trong quá trình train:

* `loss`
* `val_loss`
* `accuracy`
* `val_accuracy`

File:

```text
results/custom_cnn/training_log.csv
```

---

# 6. Thực nghiệm trên Notebook

File:

```text
02_custom_cnn.ipynb
```

được sử dụng để chạy thử model và xem kết quả trực quan.

## 6.1. Kiểm tra model

Có thể sử dụng:

```python
model.summary()
```

để kiểm tra kiến trúc và số lượng tham số.

Với tập train chỉ có 3.200 ảnh, số lượng tham số nên nằm khoảng:

```text
1.5M - 4M
```

để model không quá lớn so với dữ liệu.

---

## 6.2. Huấn luyện

Có thể đặt:

```python
epochs = 50
```

Tuy nhiên, nhờ `EarlyStopping`, model có thể tự dừng trước khi chạy hết 50 epochs nếu validation loss không còn cải thiện.

---

## 6.3. Vẽ Learning Curves

Cần vẽ hai biểu đồ:

1. Training Loss và Validation Loss.
2. Training Accuracy và Validation Accuracy.

Lưu biểu đồ tại:

```text
results/custom_cnn/learning_curves.png
```

Khi xem biểu đồ cần chú ý đến overfitting.

Ví dụ:

```text
Training Loss ↓
Validation Loss ↑
```

Nếu training loss tiếp tục giảm nhưng validation loss lại tăng thì có thể model đang bị overfitting.

---

# 7. Đánh giá trên tập Test

Sau khi train xong, load lại model tốt nhất:

```text
models/custom_cnn_best.keras
```

Sau đó đánh giá trên 800 ảnh của tập test.

Không sử dụng tập train hoặc validation để lấy kết quả đánh giá cuối cùng.

## 7.1. Classification Report

Cần tính:

* Accuracy.
* Precision.
* Recall.
* F1-score.
* Macro Average F1.
* Weighted Average F1.

Các lớp:

```text
fire
nofire
smoke
smokefire
```

---

## 7.2. Confusion Matrix

Tạo confusion matrix kích thước `4x4`.

Có thể tạo hai loại:

* Ma trận số lượng.
* Ma trận normalized theo tỷ lệ.

Sau đó vẽ heatmap và lưu:

```text
results/custom_cnn/confusion_matrix.png
```

---

# 8. Phân tích lỗi

Sau khi đánh giá model, cần xem thêm những trường hợp model dự đoán sai.

## 8.1. Các lỗi nghiêm trọng

Có hai trường hợp cần chú ý nhất:

### `fire → nofire`

Ảnh thực tế có lửa nhưng model lại dự đoán là không có cháy.

### `smokefire → nofire`

Ảnh có cả lửa và khói nhưng model lại dự đoán là ảnh bình thường.

Hai trường hợp này cần được thống kê riêng.

---

## 8.2. Các lỗi giữa những lớp gần nhau

Một số trường hợp khác:

```text
fire ↔ smokefire
```

và:

```text
smoke ↔ smokefire
```

Các lỗi này có thể xảy ra vì ảnh khó phân biệt, ví dụ lửa bị che hoặc khói quá dày.

---

## 8.3. Hiển thị ảnh bị dự đoán sai

Cần tìm các ảnh thuộc nhóm lỗi quan trọng và hiển thị dưới dạng grid.

Mỗi ảnh nên có:

* Ảnh gốc.
* Ground Truth.
* Predicted Label.
* Confidence score.

Lưu kết quả tại:

```text
results/custom_cnn/critical_errors_samples.png
```

---

# 9. File metrics.json

Sau khi đánh giá xong, lưu kết quả vào:

```text
results/custom_cnn/metrics.json
```

Ví dụ:

```json
{
    "model_name": "Custom Complex CNN",
    "test_accuracy": 0.0,
    "test_macro_f1": 0.0,
    "critical_fire_as_nofire": 0,
    "critical_smokefire_as_nofire": 0,
    "total_parameters": 0
}
```

Các giá trị `0.0` và `0` sẽ được thay bằng kết quả thực tế sau khi chạy model.

File này sẽ được sử dụng để tổng hợp kết quả vào:

```text
results/final_comparison.csv
```

---

# 10. Cấu trúc thư mục

| File                  | Chức năng                                             |
| --------------------- | ----------------------------------------------------- |
| `README.md`           | Mô tả các bước thực hiện                              |
| `model.py`            | Xây dựng kiến trúc CNN, SE Block và Residual-SE Block |
| `train.py`            | Train model, callbacks và lưu checkpoint              |
| `evaluate.py`         | Đánh giá model và phân tích lỗi                       |
| `02_custom_cnn.ipynb` | Chạy thực nghiệm và vẽ biểu đồ                        |

