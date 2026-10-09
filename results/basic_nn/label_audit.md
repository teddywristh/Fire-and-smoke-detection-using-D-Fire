> Tài liệu lịch sử của phiên bản dùng module Python. Code hiện tại đã chuyển sang [các notebook](../../notebooks/01_basic_nn.ipynb); các tham chiếu `src/` bên dưới mô tả phiên bản trước khi chuyển.

# Rà soát nhãn smoke / smokefire

## Kết luận

Không tìm thấy lỗi đảo index hoặc tách rời ảnh và nhãn trong code hiện tại.
Mapping là fire=0, nofire=1, smoke=2, smokefire=3. Notebook gọi trực tiếp
`src.basic_nn`, không có pipeline gán nhãn riêng.

Accuracy đã lưu là 478/800 = 59,75%. Trong confusion matrix:

| Nhãn thật | Dự đoán sai | Số ảnh |
|---|---|---:|
| smokefire | smoke | 98/200 |
| smoke | smokefire | 1/200 |
| fire | smokefire | 60/200 |
| smoke | nofire | 46/200 |

Lỗi smoke/smokefire bất đối xứng, không giống một phép đảo hai nhãn.
Nếu chỉ hoán đổi hai output smoke/smokefire của matrix hiện tại, accuracy
giảm xuống 340/800 = 42,5%. Không nên sửa mapping để tăng điểm.

## Kiểm tra code và manifest

- `config.py`: mapping duy nhất, phù hợp manifest.
- `src/data_loader.py`: nhãn lấy từ thư mục cha; regex khớp toàn bộ tên file,
  không dùng phép kiểm tra substring `smoke` có thể bắt nhầm `smokefire`.
- 4.800 dòng manifest: label/index và thư mục raw/processed nhất quán;
  đủ 800 ảnh/lớp train, 200 ảnh/lớp val và test; không thiếu ảnh processed.
- Đã tính lại SHA256 toàn bộ 4.800 raw và 4.800 processed: không thiếu file,
  không khác hash manifest; không có nhóm raw hash trùng giữa split/nhãn.
- `src/basic_nn.py:load_data`: kiểm tra label/index trước khi train.
- `make_dataset`: ảnh và nhãn được tạo thành cùng một tuple rồi mới shuffle;
  val/test không shuffle, không augmentation. Không thấy lệch thứ tự predict.
- `evaluate`: argmax trên trục lớp; confusion matrix, report và tên prediction
  dùng cùng mapping; đường chéo matrix khớp accuracy đã lưu.
- `src/preprocessing.py`: đổi RGB, EXIF, resize và JPEG; không thay nhãn.
- `evaluation.py`, `visualization.py`, `gradcam.py` chỉ là placeholder;
  augmentation thực tế nằm trong `basic_nn.py`, không phải policy mô tả
  trong `augmentation.py`.

## Vấn đề dữ liệu quan sát được

Xem [ảnh lỗi có confidence cao](label_audit_samples.jpg) và
[đối chiếu raw / processed](raw_processed_audit.jpg).

- `smokefire_test_1064.jpg`: ảnh raw đã có vùng kéo sọc/méo; không thấy ngọn
  lửa rõ trong ảnh được xem. Model dự đoán smoke với confidence khoảng 0,83.
- `smoke_test_1197.jpg`: vùng dưới ảnh raw đã kéo sọc; preprocessing chỉ
  resize lại ảnh lỗi sẵn có.
- Các mẫu smokefire như 1110, 1161, 1104, 1153, 1162, 1051, 1058 cần kiểm
  tra thủ công dấu hiệu lửa; thumbnail phần lớn thể hiện khói nổi bật.
- Một số mẫu fire như 1168, 1072, 1066, 1041 có dấu hiệu khói cùng lửa,
  cần đối chiếu với định nghĩa lớp trong DATA_PROCESSING.md.

Đây là quan sát trên mẫu, chưa phải xác nhận sai nhãn toàn bộ dataset.
`status=ok` chỉ xác nhận các kiểm tra kỹ thuật hiện có, không xác nhận nội
dung ảnh đúng nhãn, không méo hoặc không gần trùng.

## Giới hạn mô hình và split

Basic NN resize 224 xuống 64 rồi Flatten -> Dense(256) -> Dropout(0,5).
Chi tiết lửa nhỏ có thể mất khi giảm kích thước; mạng Dense không có cấu
trúc convolution để học đặc trưng cục bộ. Đây là yếu tố có thể góp phần
vào smokefire -> smoke, chưa được kiểm chứng bằng thí nghiệm đối chứng.

Best validation loss trong history hiện tại nằm ở epoch 30, val accuracy
72,5%, trong khi test 59,75%: chênh 12,75 điểm phần trăm. Cần rà chất lượng
ảnh và phân bố cảnh giữa các split; chưa thể kết luận chỉ từ chênh lệch này.
Các lớp cân bằng nên thiếu số lượng smoke/smokefire không giải thích lỗi.
Kích thước raw tương quan với lớp; direct resize còn làm méo ảnh không vuông.

## Hướng xử lý

1. Rà ảnh raw trên cả train/val/test theo tiêu chí thống nhất: có lửa,
   có khói, cả hai, hoặc không có cả hai. Đánh dấu ảnh mơ hồ và ảnh kéo sọc;
   ghi nhãn cũ, nhãn đề xuất, lý do và người duyệt trước khi sửa.
2. Thay ảnh lỗi bằng bản nguồn tốt nếu có; không tự relabel theo prediction.
   Kiểm tra ảnh gần trùng/cùng cảnh giữa split, SHA256 chỉ bắt trùng chính xác.
3. Giữ Basic NN làm baseline. So sánh kích thước đầu vào và CNN/transfer
   learning bằng validation sau khi làm sạch dữ liệu.
4. Lưu class_names cùng checkpoint và kiểm tra khi load, để ngăn việc thay
   mapping ở lần chạy sau. Mapping của checkpoint hiện chưa có metadata này.
5. Chọn cấu hình trên validation; dùng holdout mới nếu test đã được dùng
   nhiều lần để lựa chọn cấu hình như tài liệu lịch sử ghi nhận.

Không đổi nhãn, không train lại và không ghi đè kết quả hiện có trong lần
rà soát này. Các số accuracy lấy từ artifacts đã lưu; chưa chạy lại checkpoint.
