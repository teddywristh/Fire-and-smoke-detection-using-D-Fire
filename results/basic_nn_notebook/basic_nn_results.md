# Kết quả Basic NN từ notebook

Chỉ sử dụng mạng Dense, toàn bộ code nằm trong notebook 01. Cấu hình mới:
Dense(256) → BatchNormalization → ReLU → Dropout(0,3) → Dense(128, ReLU)
→ Dropout(0,2) → Softmax(4). Ảnh 64×64, antialias, lật ngang train,
Adam 0,0003, batch 32, seed 42.

Đã chạy 23 epoch, checkpoint theo validation loss tại epoch 18. Validation
accuracy tại checkpoint là 75,125%, validation loss 0,6407.
Giữ nguyên nhãn và 800 ảnh test.

| Chỉ số | Baseline cũ | Basic NN mới |
|---|---:|---:|
| Accuracy | 59,75% | 60,25% |
| Macro F1 | 0,6006 | 0,5928 |
| smokefire → smoke | 98 | 82 |
| smoke → smokefire | 1 | 10 |
| Số ảnh dự đoán sai | 322 | 318 |

Accuracy tăng 0,5 điểm phần trăm (thêm 4 ảnh đúng), nhưng Macro F1 giảm.
Một lần chạy chưa đủ kết luận cải thiện ổn định. Không dùng test để tiếp
tục chọn cấu hình hoặc thay nhãn. Ảnh raw méo/kéo sọc và nhãn mơ hồ vẫn
cần duyệt dữ liệu; tinh chỉnh Dense chưa giải quyết hết các hạn chế này.

Baseline cũ được giữ riêng. Kết quả mới có CSV/JSON, ảnh lỗi và biểu đồ,
đồng thời hiển thị trực tiếp trong notebook 01. Accuracy từ confusion
matrix đã được kiểm tra khớp model.evaluate.
