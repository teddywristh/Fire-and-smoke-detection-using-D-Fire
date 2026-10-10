# Kiến trúc nhánh Frozen Transfer Learning

Tài liệu này mô tả pipeline so sánh hai backbone chính: ResNet-50 và Xception. Code dùng chung nằm trong [`src/training/xception_aspect_frozen.py`](../src/training/xception_aspect_frozen.py); kết quả validation và các giới hạn lịch sử được tổng hợp tại [`frozen_features.md`](frozen_features.md).

## 1. Mục tiêu và phạm vi

Nhánh frozen giữ nguyên trọng số backbone pretrained ImageNet và chỉ huấn luyện classifier head. Hai backbone được chạy trên cùng `data/split_aspect.csv`, cùng protocol crop, sampler, head, optimizer, checkpoint rule và bốn seed (`42`, `7`, `123`, `2024`). Pipeline chỉ load train và validation; không load ảnh hoặc feature của test.

Đây là so sánh validation-only. Test gốc từng được dùng trong các thử nghiệm lịch sử, vì vậy các metric test cũ không phải đánh giá độc lập cuối cùng.

## 2. Luồng xử lý

```text
data/split_aspect.csv
        │
        ├── train ── đọc ảnh gốc ── square crop ngẫu nhiên ── augmentation
        │                                          │
        │                                   scale pixel [-1, 1]
        │                                          │
        │                          frozen pretrained backbone
        │                                          │
        │                               cache feature train
        │                                          │
        │                          aspect-balanced head training
        │
        └── val ──── đọc ảnh gốc ── center square crop ── resize
                                                   │
                                           scale pixel [-1, 1]
                                                   │
                                    frozen backbone + feature cache
                                                   │
                               chọn checkpoint theo validation macro F1
```

Manifest chứa test membership để tái sử dụng split, nhưng script chỉ giữ các hàng `train` và `val` trước khi kiểm tra đường dẫn ảnh và tạo cache.

## 3. Backbone và classifier head

| Backbone | Tên model trong `timm` | Trạng thái khi huấn luyện |
| --- | --- | --- |
| ResNet-50 | `resnet50` | Backbone frozen; chỉ classifier head có gradient |
| Xception | `legacy_xception` | Backbone frozen; chỉ classifier head có gradient |

Cả hai backbone xuất vector pooled 2048 chiều cho cấu hình hiện tại. Head dùng chung:

```text
2048
  → Dropout(0.45)
  → Linear(512)
  → ReLU
  → BatchNorm1d(512)
  → Dropout(0.20)
  → Linear(4)
```

Đầu ra có bốn logit theo thứ tự lớp `fire`, `nofire`, `smoke`, `smokefire`; loss là `CrossEntropyLoss`.

## 4. Tiền xử lý và augmentation trong protocol so sánh

Pipeline đọc ảnh gốc theo `filepath` trong manifest, áp dụng EXIF orientation và chuyển RGB.

| Split | Crop và resize | Augmentation |
| --- | --- | --- |
| Train | Square crop ngẫu nhiên, sau đó resize 224×224 bằng bilinear | Affine nhẹ, lật ngang ngẫu nhiên, brightness/contrast jitter |
| Validation | Center square crop, sau đó resize 224×224 bằng bilinear | Không có |

Square crop chọn cạnh không lớn hơn cạnh ngắn của ảnh; train chọn ngẫu nhiên một phần cạnh từ 72% đến 100% và vị trí crop, validation lấy crop giữa. Augmentation train dùng góc ±5°, translate ±4%, scale 0.94–1.06, shear ±2°, flip xác suất 0.5 và brightness/contrast 0.88–1.12.

Ảnh được đổi từ pixel `[0, 255]` sang tensor float `[-1, 1]`. **Lưu ý triển khai:** script đọc mean/std từ `pretrained_cfg`, nhưng wrapper đầu vào hiện tại không áp dụng các giá trị đó. Vì vậy tensor thực sự đưa vào cả hai backbone đang ở `[-1, 1]`; không nên mô tả protocol này là áp dụng ImageNet mean/std riêng cho từng backbone. Thay đổi phép chuẩn hóa sẽ tạo protocol khác và cần chạy lại các thí nghiệm mới trước khi so sánh.

Một view train được augmentation và feature-cache một lần cho mỗi run; head sau đó huấn luyện trên feature đã cache, không sinh lại crop/augmentation ở từng epoch.

## 5. Aspect-balanced sampling

Aspect group được tính từ tỷ lệ `width / height`: `square` trong `[0.90, 1.10]`, `wide` trong `[1.60, 1.95]`, các ảnh còn lại là `other`. Với mỗi cặp `(class, aspect group)`, trọng số mẫu tỷ lệ nghịch với căn bậc hai số lượng phần tử của nhóm:

```text
weight(sample) = 1 / sqrt(count(class, aspect_group))
```

Mỗi epoch lấy số mẫu bằng kích thước tập train, có hoàn lại, bằng `torch.multinomial`. Sampling này giảm ảnh hưởng của các nhóm class/aspect hiếm; nó không làm tăng lượng dữ liệu độc lập trong nhóm đó.

## 6. Huấn luyện và chọn checkpoint

- Optimizer: AdamW, learning rate `5e-4`, weight decay `1e-4`.
- Scheduler: `ReduceLROnPlateau` theo validation loss, factor `0.5`, patience 2.
- Early stopping: patience 6 epoch.
- Checkpoint: validation macro F1 cao nhất; nếu bằng nhau, chọn validation loss thấp hơn.
- Seeds: `42`, `7`, `123`, `2024` cho từng backbone.

Ví dụ chạy một seed từ thư mục gốc project:

```powershell
python -m src.training.xception_aspect_frozen --architecture resnet50 --split-csv data/split_aspect.csv --crop-mode square --aspect-balance --image-size 224 --epochs 30 --batch-size 32 --cpu-threads 4 --seed 42 --run-name resnet50_square_aspect_balanced_224_seed_42
```

Đổi `--architecture` thành `xception`, và thay seed/run name cho mỗi lần chạy. Cần cấu hình `DATASET_ROOT` cục bộ để đọc ảnh gốc. Lệnh không chạy lại training khi mở notebook; notebook mặc định chỉ đọc CSV artifacts.

## 7. Artifacts

Artifacts so sánh được lưu riêng:

- `results/frozen_backbone_comparison_per_seed.csv`: một hàng cho mỗi backbone/seed, gồm accuracy, macro F1 và F1 của từng lớp.
- `results/frozen_backbone_comparison_mean_variance.csv`: mean, population variance (`ddof=0`) và số seed cho các metric.
- Các thư mục con theo từng run trong `results/transfer_frozen/`: report, confusion matrix, training history và aspect audit. ResNet-50 có tên thư mục kèm từng seed; Xception seed 42 nằm trong `xception_square_aspect_balanced_224/`, các seed còn lại có hậu tố `_seed_{7,123,2024}`.
- `data/cache/`: feature cache; `models/transfer_frozen/`: checkpoint.

CSV comparison và manifest là artifacts cần thiết để tái tạo bảng/biểu đồ. Không cần commit raw dataset, ảnh processed, feature cache hoặc checkpoint để xem các kết quả đã lưu. Run directories có thể chứa đường dẫn máy cục bộ trong metadata.

## 8. Kết quả validation qua bốn seed

| Backbone | Accuracy mean | Accuracy variance | Macro F1 mean | Macro F1 variance |
| --- | ---: | ---: | ---: | ---: |
| Xception | 88.94% | 0.000101953 | 88.96% | 0.000094932 |
| ResNet-50 | 93.34% | 0.000015137 | 93.34% | 0.000014739 |

Tính theo mean từ các CSV, ResNet-50 cao hơn khoảng 4.41 điểm phần trăm accuracy và 4.38 điểm phần trăm macro F1. Variance thấp hơn cho thấy metric ít dao động hơn giữa bốn seed trong protocol này; không đảm bảo kết quả tương tự trên nguồn ảnh hoặc split khác.

## 9. Diễn giải và giới hạn

- Kết quả trên là validation-only; không phải kết luận độc lập trên test.
- Macro F1 checkpoint selection dùng validation, nên metric validation cũng tham gia quá trình chọn model.
- Validation có ít ảnh `smokefire` wide; không thể dùng riêng nhóm này để kết luận độ bền với camera wide.
- Do test gốc đã được xem trong các thí nghiệm cũ, đánh giá cuối đáng tin cậy cần một test set độc lập mới.
- Nếu đổi manifest, crop, augmentation, input scaling, sampler, head, optimizer hoặc checkpoint rule, cần ghi thành protocol mới; không gộp các run đó vào CSV comparison hiện tại.

## 10. Tài liệu liên quan

- [Preprocessing dữ liệu](data_preprocessing.md)
- [Lịch sử nhánh frozen features](frozen_features.md)
- [Notebook so sánh](../notebooks/frozen_transfer_learning_pipeline.ipynb)
