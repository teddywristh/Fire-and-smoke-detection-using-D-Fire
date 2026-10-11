# So sánh frozen ResNet-50 và Xception

## Mục tiêu

Báo cáo này so sánh **hai backbone đều frozen**: ResNet-50 và Xception. Không đưa kết quả fine-tuned vào phép so sánh. Số liệu được tổng hợp từ artifacts hiện có, không huấn luyện lại model hoặc chạy inference lại.

## Giao thức

- Manifest: `data/split_aspect.csv`; chỉ dùng train và validation.
- Input: square crop, resize 224×224.
- Sampling: cân bằng theo cặp `(class, aspect group)`.
- Backbone: trọng số ImageNet giữ frozen; chỉ classifier head được huấn luyện.
- Head, optimizer và quy tắc chọn checkpoint dùng chung; checkpoint chọn theo validation macro F1, validation loss dùng làm tie-breaker.
- Seeds cho mỗi model: `42`, `7`, `123`, `2024`.
- Test set không được load trong các run so sánh này.

Chi tiết kiến trúc và preprocessing, gồm lưu ý về input scaling, nằm trong [tài liệu kiến trúc frozen](frozen_transfer_learning_architecture.md). Protocol này khác baseline lịch sử trong `results/transfer_frozen/resnet50/summary.md`, vốn dùng `data/split.csv`, direct resize và checkpoint theo validation loss. Vì vậy không gộp hoặc so trực tiếp hai loại kết quả đó.

## Kết quả từng seed

| Seed | Xception accuracy | Xception macro F1 | ResNet-50 accuracy | ResNet-50 macro F1 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 90.250% | 90.256% | 93.625% | 93.620% |
| 7 | 88.625% | 88.637% | 93.250% | 93.226% |
| 123 | 89.375% | 89.345% | 93.750% | 93.743% |
| 2024 | 87.500% | 87.594% | 92.750% | 92.761% |

## Mean và population variance

Variance được tính trên bốn giá trị theo seed với `ddof=0`. Mean là phần trăm; variance được giữ trên thang metric từ 0 đến 1.

| Metric | Xception mean | Xception variance | ResNet-50 mean | ResNet-50 variance | ResNet-50 trừ Xception |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 88.938% | 0.000101953 | 93.344% | 0.000015137 | +4.406 điểm % |
| Macro F1 | 88.958% | 0.000094932 | 93.338% | 0.000014739 | +4.380 điểm % |

## Mean và population variance của F1 theo lớp

Variance dưới đây lấy trực tiếp từ `results/frozen_backbone_comparison_mean_variance.csv`
với `ddof=0`, cùng cách tính được sử dụng trong notebook.

| Lớp | Xception mean | Xception variance | ResNet-50 mean | ResNet-50 variance | Chênh lệch mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fire` | 87.011% | 0.000159946 | 89.082% | 0.000083435 | +2.071 điểm % |
| `nofire` | 94.011% | 0.000092633 | 98.178% | 0.000004222 | +4.167 điểm % |
| `smoke` | 92.249% | 0.000052670 | 97.953% | 0.000018358 | +5.704 điểm % |
| `smokefire` | 82.560% | 0.000164830 | 88.137% | 0.000030552 | +5.577 điểm % |

## Trực quan hóa trong notebook

Notebook [`frozen_transfer_learning_pipeline.ipynb`](../notebooks/frozen_transfer_learning_pipeline.ipynb)
đọc các CSV kết quả và dựng ba nhóm biểu đồ:

- accuracy theo từng seed cho hai backbone;
- macro F1 theo từng seed cho hai backbone;
- F1 trung bình theo lớp để quan sát lớp khó, đặc biệt `smokefire`.

Các biểu đồ này chỉ trực quan hóa validation metrics của protocol frozen đã khóa,
không phải kết quả test và không thay thế đánh giá trên test độc lập.

Trong bốn lần chạy này, ResNet-50 có mean accuracy và macro F1 cao hơn; population variance của hai metric cũng thấp hơn. F1 trung bình theo lớp cao hơn ở cả bốn lớp.

## Diễn giải đúng mức

Đây là kết quả **validation-only** trong một protocol đã khóa. Test gốc không được dùng trong các run so sánh này, nhưng trước đó đã được quan sát trong những thử nghiệm lịch sử; vì vậy kết quả ở đây không phải đánh giá độc lập trên test set.

Bốn seed chạy trên cùng validation examples. Population variance cho biết độ dao động do seed trong các run này; nó **không phải** confidence interval hoặc ước lượng biến thiên khi lấy mẫu một tập dữ liệu mới. CSV hiện có metric tổng hợp theo run, không có dự đoán từng ảnh để tính kiểm định ghép cặp hoặc khoảng tin cậy theo ảnh. Do đó báo cáo không khẳng định p-value hay ý nghĩa thống kê.

Model selection và đánh giá cần được tách biệt để tránh selection bias; nested cross-validation là lựa chọn nội bộ nếu không có test mới, nhưng kết quả vẫn phải trình bày là ước lượng trên dữ liệu hiện có, không phải external test. Xem Cawley và Talbot về selection bias trong đánh giá model ([JMLR, 2010](https://www.jmlr.org/papers/volume11/cawley10a.html)) và Varoquaux về độ bất định của cross-validation ([NeuroImage, 2018](https://pubmed.ncbi.nlm.nih.gov/28655633/)).

## Artifacts và mã nguồn

- `results/frozen_backbone_comparison_per_seed.csv`: metric từng seed, gồm F1 từng lớp.
- `results/frozen_backbone_comparison_mean_variance.csv`: mean, population variance và số seed.
- [`notebooks/frozen_transfer_learning_pipeline.ipynb`](../notebooks/frozen_transfer_learning_pipeline.ipynb): bảng và biểu đồ được dựng từ CSV.
- [`src/training/xception_aspect_frozen.py`](../src/training/xception_aspect_frozen.py): pipeline dùng chung, chọn backbone qua `--architecture`.

## Tài liệu liên quan

- [Preprocessing dữ liệu](data_preprocessing.md)
- [Kiến trúc nhánh frozen transfer learning](frozen_transfer_learning_architecture.md)
- [Lịch sử các thí nghiệm frozen](frozen_features.md)
