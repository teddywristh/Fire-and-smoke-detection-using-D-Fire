# Xử lý dữ liệu

Toàn bộ pipeline nằm trong [notebook 00](notebooks/00_data_preparation.ipynb).
Chạy các cell từ trên xuống; sửa `DATASET_ROOT` ngay trong cell cấu hình.
Không dùng `local_config.py`, biến môi trường hoặc module `src` nữa.

## Quy trình

1. Duyệt train/val/test và bốn lớp; kiểm tra tên file khớp thư mục.
2. Đọc ảnh, áp dụng EXIF orientation, chuyển RGB, direct resize 224×224.
3. Lưu JPEG chất lượng 95 và metadata/SHA256 trong `data/split.csv`.
4. Đánh dấu file lỗi hoặc trùng chính xác bằng `status=review`.
5. Xem mẫu từng lớp/tập, tạo danh sách cần duyệt từ các lỗi dự đoán đã lưu.

Mapping: fire=0, nofire=1, smoke=2, smokefire=3.
Các notebook mô hình chỉ lấy `status=ok` và kiểm tra mapping trước khi chạy.
Không tự relabel theo prediction hoặc loại các ảnh test khó để tăng accuracy.

## Điều kiểm tra tự động không xác nhận được

`status=ok` không đồng nghĩa ảnh đúng nhãn hoặc không bị méo. SHA256 chỉ phát
hiện file trùng chính xác, không bắt hết cảnh giống nhau hoặc ảnh gần trùng.
Direct resize có thể biến dạng ảnh không vuông; ảnh kéo sọc đã có trong raw
cần được thay bằng bản nguồn tốt, không thể sửa chỉ bằng thay interpolation.

`data/label_review_candidates.csv` lưu nhãn hiện tại, gợi ý và quyết định để
duyệt thủ công. File này không tự thay đổi dữ liệu huấn luyện/đánh giá.
Nếu thay nhãn hoặc loại ảnh, cần lưu bộ dữ liệu phiên bản mới và báo cáo rõ
thay đổi số ảnh; kết quả mới không còn so sánh trực tiếp với test cũ.

## Đầu vào Basic NN

Notebook 01 dùng ảnh processed, resize 64×64 và chuẩn hóa pixel [0, 1].
Augmentation chỉ áp dụng cho train. Giữ nguyên split và mapping.
Notebook 00 chỉ hỗ trợ chuẩn bị dữ liệu và rà nhãn cho Basic NN.
