# Basic Neural Network — Phân loại cháy rừng và khói

Hiện tại dự án chỉ tập trung phát triển **Basic NN dùng Dense**.
Toàn bộ code nằm trong notebook, ghi chú tiếng Việt có dấu rồi đến code.
Không có class tự viết; notebook mô hình chỉ có một hàm đọc ảnh.

## Cách chạy

Cài thư viện: `python -m pip install -r requirements.txt`.
Chọn kernel Python có các thư viện đó, chạy cell từ trên xuống.

1. [00 — Chuẩn bị dữ liệu](notebooks/00_data_preparation.ipynb): chỉ cần khi
   chưa có manifest/ảnh processed hoặc đổi dữ liệu. Sửa `DATASET_ROOT` ở cell đầu.
2. [01 — Basic NN](notebooks/01_basic_nn.ipynb): sáu bước import → dữ liệu
   → model/compile → train → evaluate → báo cáo.

Notebook 01 mặc định `TRAIN_MODEL=True`. Đổi False để đánh giá checkpoint đã lưu.
Baseline cũ được giữ ở `models/basic_nn_best.keras` và `results/basic_nn/`.
Bản mới lưu riêng ở `models/basic_nn_notebook.keras` và `results/basic_nn_notebook/`.

## Dữ liệu và mô hình

Mapping: fire=0, nofire=1, smoke=2, smokefire=3.
Giữ nguyên 3.200 train, 800 validation, 800 test; bốn lớp cân bằng.
Ảnh RGB 64×64, pixel [0, 1]. Train shuffle và lật ngang; val/test giữ thứ tự.

Basic NN mới thử Dense(256) → BatchNormalization → ReLU → Dropout(0,3)
→ Dense(128, ReLU) → Dropout(0,2) → Softmax(4).
Adam 0,0003, tối đa 30 epoch, checkpoint chọn bằng validation loss.
Nhãn số nguyên dùng sparse cross-entropy, không cần one-hot.

## Kết quả và giới hạn

Baseline cũ đạt 59,75%. Kết quả bản mới được đo thực tế trong notebook 01 và
`results/basic_nn_notebook/`, gồm history, metrics, predictions, F1,
confusion matrix và biểu đồ. Không dùng test để chọn epoch hoặc tuning tiếp.

Bản Dense mới đạt **60,25%**, tăng 0,5 điểm phần trăm; Macro F1 giảm từ
0,6006 xuống 0,5928. Đây là thay đổi nhỏ của một seed, chưa chứng minh cải thiện
ổn định. Xem [kết quả Basic NN](results/basic_nn_notebook/basic_nn_results.md).

[Rà soát nhãn](results/basic_nn/label_audit.md) có các mẫu raw kéo sọc/méo
và nhãn cần duyệt lại. `data/label_review_candidates.csv` chỉ hỗ trợ rà nhãn,
không tự đổi nhãn hoặc loại ảnh test khó. Mạng Dense không thể khôi phục
ảnh hỏng hoặc tự xác nhận nhãn đúng. Test đã được dùng trong lựa chọn lịch sử;
cần holdout mới để đánh giá độc lập.
