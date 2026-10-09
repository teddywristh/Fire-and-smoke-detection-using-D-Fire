> Tài liệu lịch sử của phiên bản dùng module Python. Code hiện tại đã chuyển sang [các notebook](../../notebooks/01_basic_nn.ipynb); các tham chiếu `src/` bên dưới mô tả phiên bản trước khi chuyển.

# Phương pháp và cách làm Basic Neural Network

> **Integration note (2026-10-06).** The code moved from `src/basic_nn.py` to
> `src/models/basic_nn.py` and now reads data from the shared pipeline
> (`src/datasets.py`). Architecture and training settings are unchanged. Two
> things differ from the run recorded in this folder:
>
> - Augmentation is now the shared policy in `src/augmentation.py` (flip,
>   rotation ±10°, shift and zoom 10%, brightness ×0.8–1.2) instead of flip,
>   brightness ±0.08 and contrast 0.9–1.1 (section 3).
> - Resizing to 64×64 and scaling to [0, 1] now happen inside the model
>   (`Resizing` and `Rescaling` layers), and labels are one-hot with
>   categorical cross-entropy, which gives the same loss as before.
>
> The numbers in `basic_nn_results.md` come from the old pipeline. Re-run
> `python -m src.models.basic_nn` to get results on the shared pipeline.
> Checkpoints from the old pipeline are rejected by `--evaluate-only`.

## 1. Mục tiêu và phạm vi

Phân loại ảnh thành bốn lớp `fire`, `nofire`, `smoke`, `smokefire` bằng một
mạng fully connected làm baseline. Toàn bộ code Basic NN nằm trong
`src/models/basic_nn.py`: đọc dữ liệu → tạo model → compile → fit → evaluate/predict
→ lưu metrics, biểu đồ và báo cáo kết quả.

Giữ một cấu hình và một checkpoint. [Kết quả chạy](basic_nn_results.md) nằm
trong file riêng; tài liệu này giải thích phương pháp và cách thực hiện.

## 2. Đối chiếu với slide môn học

Đã đọc 12 bộ slide được cung cấp. Các bài tập, ví dụ và yêu cầu trong slide
được dùng để hiểu phương pháp, không được coi là yêu cầu thay đổi bài toán
forest-fire-c4 thành Wine, Cars, Bank Marketing, MNIST hoặc CIFAR-10.

| Nội dung học | Áp dụng vào dự án |
|---|---|
| [Chương 1: Introduction](<../../../Chapter 1-Introduction.pdf>) | Bài toán supervised classification, dùng nhãn có sẵn và TensorFlow |
| [Chương 2.1: Linear Regression](<../../../Chapter 2.1-Linear Regression.pdf>) | Dense thực hiện tổng có trọng số và bias; không dùng MSE của hồi quy cho bài toán bốn lớp |
| [Chương 2.2: Gradient Descent](<../../../Chapter 2.2-Gradient Descent.pdf>) | Mini-batch 32, shuffle train, learning rate; dùng gradient để cập nhật trọng số |
| [Chương 3.1: Logistic Regression](<../../../Chapter 3.1-Logistic Regression.pdf>) | Liên hệ đầu ra xác suất; bài toán này cần Softmax bốn lớp thay vì một sigmoid nhị phân |
| [Chương 3.2: Softmax Regression](<../../../Chapter 3.2 Softmax Regression.pdf>) | Softmax, cross-entropy và chọn lớp bằng argmax, nhất là trang 5, 9, 14 |
| [Chương 4.1: Neural Network](<../../../Chapter 4.1 - Neural Network.pdf>) | Feedforward, ReLU, backpropagation; Keras tự tính gradient trong fit |
| [Chương 4.2: Computing](<../../../Chapter 4.2-Computing in deep learning.pdf>) | Sequential và Dense(256, ReLU), tương ứng ví dụ trang 4; dùng Keras thay MXNet |
| [Chương 4.3: Optimal Training](<../../../Chapter 4.3-Optimal Training.pdf>) | ReLU và Dropout để hỗ trợ huấn luyện/regularization, trang 5 và 15; BN/L2 là lựa chọn, không bắt buộc thêm mọi kỹ thuật |
| [Bài tập Chương 4](<../../../Chapter 4_Neural Networks_Exercises.pdf>) | TensorFlow, fit/predict, Accuracy/Precision/Recall/F1 và mapping nhãn thống nhất; giữ split gốc của dataset này |
| [Chương 5.1: CNN](<../../../Chapter 5.1 - Convolutional Neural Network ‐ CNN.pdf>) | Phong cách model → compile → fit → evaluate tại trang 39–42; Adam trong ví dụ trang 40 |
| [Chương 5.2: Modern CNN](<../../../Chapter 5.2-Modern convolutional neural networks.pdf>) | Hiểu các kiến trúc CNN/BN; không đưa VGG/ResNet vào baseline Basic NN |
| [Chương 5.3: CNN technique](<../../../Chapter 5.3 - CNN technique.pdf>) | Augmentation nhẹ, trang 13–14; transfer learning là phương pháp tiếp theo, không trộn vào Basic NN |

Phương pháp hiện tại phù hợp nội dung học. Không cần tự viết lớp Block,
backpropagation hoặc optimizer để sử dụng TensorFlow theo phần I của bài tập.
Phần II yêu cầu tự xây Neural Network class là bài tập riêng, không phải điều
kiện bắt buộc của mọi project.

## 3. Dữ liệu và tiền xử lý

Nguồn: Forest Fire Image Classification Dataset (forest-fire-c4).
Manifest `data/split.csv`, chỉ dùng dòng `status=ok`.
Giữ nguyên 3.200 ảnh train, 800 validation, 800 test; mỗi split cân bằng bốn lớp.
Mapping: `fire=0`, `nofire=1`, `smoke=2`, `smokefire=3`.

Pipeline chung lưu ảnh RGB JPEG 224×224 bằng direct resize. Basic NN resize
runtime về 64×64 và chia pixel cho 255. Đầu vào Flatten còn 12.288 đặc trưng,
thay vì 150.528 nếu giữ 224×224. Resize trực tiếp có thể làm méo ảnh không vuông;
đây là chính sách đang dùng chung, không phải kết luận bắt buộc từ slide.

Train shuffle và augmentation runtime: lật ngang, brightness ±0.08,
contrast 0.9–1.1, clip [0, 1]. Validation/test không augmentation, không shuffle.
`Forest Fire_Tester` chỉ dùng demo. Không đưa ảnh test vào train.

## 4. Model và huấn luyện

```text
Input(64, 64, 3)
→ Flatten
→ Dense(256, ReLU)
→ Dropout(0.5)
→ Dense(4, Softmax)
```

- 3.147.012 tham số; một hidden layer, không Conv/Pooling/Transfer Learning.
- Adam, learning rate 0.0005; batch size 32; tối đa 30 epochs; seed 42.
- Loss `sparse_categorical_crossentropy` vì nhãn là số nguyên 0–3.
- ModelCheckpoint và EarlyStopping dùng `val_loss`, patience 5.
- ReduceLROnPlateau dùng `val_loss`, factor 0.5, patience 2, min_lr 0.00001.
- Tải checkpoint tốt nhất rồi đánh giá test và predict theo thứ tự manifest.

Trong slide, cross-entropy dùng one-hot: `-sum(y_k * log(p_k))`.
Với nhãn số nguyên, sparse cross-entropy là `-log(p_lớp_thật)`; hai cách
tương đương về mục tiêu học. Không cần one-hot chỉ để giống cú pháp ví dụ.
Keras mặc định khởi tạo Dense bằng Glorot uniform; đây không phải khởi tạo
normal rất nhỏ bị cảnh báo trong slide. Không cần tự viết initializer.

EarlyStopping/ModelCheckpoint là tiện ích triển khai để chọn model bằng
validation, không phải thuật toán mới được yêu cầu nguyên văn trong slide.
Kiến trúc, loss, augmentation và optimizer được giữ nguyên khi rút gọn code,
do đó không coi việc rút ngắn code là một cải thiện accuracy.

## 5. Cải thiện accuracy theo cách đơn giản

1. Đánh giá train và validation trong cùng chế độ inference, không augmentation
   và không Dropout, để chẩn đoán. Train accuracy trong fit có hai yếu tố này
   nên thấp hơn val chưa chứng minh underfitting.
2. Thử từng thay đổi nhỏ: trước hết Dropout 0.5 → 0.3; giữ nguyên các yếu tố khác.
   Chỉ giữ thay đổi nếu validation tốt hơn. Mức 0.3 là giá trị thử, không phải
   con số được slide khẳng định tối ưu.
3. Nếu tối ưu chưa ổn, thử learning rate 0.0005 → 0.0003 bằng một lần chạy riêng.
   Theo dõi validation loss/accuracy và thời gian hội tụ; không giảm learning
   rate rồi mặc định cho rằng accuracy sẽ tăng.
4. Khi có dấu hiệu overfitting mới cân nhắc L2 nhỏ hoặc điều chỉnh augmentation,
   từng yếu tố một. Không thêm đồng thời BN, nhiều Dense, L2 và lịch learning rate.
5. Chọn cấu hình bằng validation loss, xem thêm validation accuracy/Macro F1;
   test dùng báo cáo cuối. Các kết quả test lịch sử là hồi cứu, không dùng để
   chọn các thử nghiệm mới. Muốn đánh giá độc lập sau lựa chọn cũ cần holdout mới.

Đã chạy riêng ba cách: Dropout 0.3, learning rate 0.0003 và ReduceLROnPlateau.
Chọn bằng validation loss nhỏ nhất: ReduceLROnPlateau với Dropout 0.5 và
learning rate ban đầu 0.0005. Sau lựa chọn, test accuracy đạt 59.75%,
tăng 3.50 điểm phần trăm so với checkpoint ngay trước thử (56.25%),
và tăng 0.375 điểm so với mốc lịch sử 59.375%. Kết quả một seed chưa chứng minh
cải thiện ổn định; các số chi tiết nằm trong file kết quả.
Basic NN Flatten không khai thác tốt cấu trúc không gian ảnh. Nếu cần cải thiện
mạnh hơn, CNN đơn giản ở Chương 5.1 là phương pháp tiếp theo phù hợp chương trình;
vẫn giữ Basic NN này làm baseline riêng.

## 6. Cách chạy

Có thể mở `notebooks/01_basic_nn.ipynb` để chạy cùng phương pháp theo sáu bước:
import/cấu hình, dữ liệu, model/compile, train, đánh giá, báo cáo. Notebook dùng
chung các hàm của `src/basic_nn.py`. `TRAIN_MODEL=True` train lại và ghi đè kết quả;
`False` chỉ xem checkpoint và kết quả đã lưu.

Tại thư mục gốc dự án:

```powershell
python -m pip install -r requirements.txt
python -m src.data_loader
python -m src.models.basic_nn
```

Bỏ bước chuẩn bị dữ liệu nếu đã có manifest và ảnh processed.
Train lại sẽ thay checkpoint và kết quả hiện tại.

```powershell
# Chỉ đánh giá checkpoint đã lưu
python -m src.models.basic_nn --evaluate-only

# Chỉ cập nhật Markdown kết quả từ CSV/JSON
python -m src.models.basic_nn --report-only
```

Checkpoint: `models/basic_nn_best.keras`. Tất cả kết quả ở `results/basic_nn/`:
metrics JSON, history/classification/confusion/errors CSV, model summary và PNG.
Tài liệu phương pháp này được giữ riêng để code không chứa hàng trăm dòng
thuyết minh. README hướng dẫn chung và DATA_PROCESSING mô tả pipeline dữ liệu.
Seed cố định hỗ trợ tái lập; phần cứng, TensorFlow và random augmentation song song
có thể khiến kết quả train lại khác nhau.
