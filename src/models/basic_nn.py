"""Basic NN: shared data pipeline -> Sequential -> compile -> fit -> evaluate.

Run: python -m src.models.basic_nn
Options: --evaluate-only (saved model), --report-only (saved results).
"""

import argparse
import json
from io import StringIO

# On Windows, import TensorFlow before numpy/sklearn/matplotlib.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import config
from src.datasets import INPUT_SHAPE, load_frames, make_dataset

INPUT_SIZE = (64, 64)
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5
MODEL_PATH = config.PROJECT_ROOT / 'models' / 'basic_nn_best.keras'
RESULT_DIR = config.PROJECT_ROOT / 'results' / 'basic_nn'


def build_model():
    # The shared pipeline gives 224x224 pixels in [0, 255]; resize and scale here.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE),
        tf.keras.layers.Resizing(*INPUT_SIZE),
        tf.keras.layers.Rescaling(1 / 255),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(DENSE_UNITS, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(len(config.CLASS_NAMES), activation='softmax'),
    ], name='basic_neural_network')
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_results(history, matrix, mistakes):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['accuracy', 'loss']):
        ax.plot(history[metric], label='train')
        ax.plot(history['val_' + metric], label='validation')
        ax.set(title=metric.title(), xlabel='Epoch (index starts at 0)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / 'accuracy_loss_curves.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.colorbar(ax.imshow(matrix, cmap='Blues'), ax=ax)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=config.CLASS_NAMES,
           yticklabels=config.CLASS_NAMES, xlabel='Predicted', ylabel='True',
           title='Basic NN Confusion Matrix')
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(RESULT_DIR / 'confusion_matrix.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, mistakes.head(16).itertuples()):
        ax.imshow(plt.imread(config.PROJECT_ROOT / row.processed_filepath))
        ax.set_title(f'True: {row.label}\nPred: {row.predicted_label} ({row.confidence:.2f})',
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / 'misclassified_samples.png', dpi=150)
    plt.close(fig)


def evaluate(model, test, settings):
    dataset = make_dataset(test, batch_size=BATCH_SIZE)
    loss, accuracy = model.evaluate(dataset, verbose=0)
    probabilities = model.predict(dataset, verbose=0)
    predicted = probabilities.argmax(axis=1)
    labels = test['class_index'].to_numpy()
    report = classification_report(labels, predicted, labels=range(4),
        target_names=config.CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.rename_axis('class').to_csv(RESULT_DIR / 'classification_report.csv')
    matrix = confusion_matrix(labels, predicted, labels=range(4))
    pd.DataFrame(matrix, index=config.CLASS_NAMES, columns=config.CLASS_NAMES).rename_axis(
        'true_label').to_csv(RESULT_DIR / 'confusion_matrix.csv')
    predictions = test[['processed_filepath', 'label', 'class_index']].copy()
    predictions['predicted_index'] = predicted
    predictions['predicted_label'] = [config.CLASS_NAMES[i] for i in predicted]
    predictions['confidence'] = probabilities.max(axis=1)
    mistakes = predictions[predictions['class_index'] != predictions['predicted_index']]
    mistakes.to_csv(RESULT_DIR / 'misclassified.csv', index=False)
    metrics = dict(test_loss=float(loss), test_accuracy=float(accuracy),
        macro_f1=report['macro avg']['f1-score'], weighted_f1=report['weighted avg']['f1-score'],
        misclassified_count=len(mistakes), parameter_count=model.count_params(),
        training_config=settings)
    (RESULT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    summary = StringIO()
    model.summary(print_fn=lambda line: summary.write(line + '\n'))
    (RESULT_DIR / 'model_summary.txt').write_text(summary.getvalue(), encoding='utf-8')
    plot_results(pd.read_csv(RESULT_DIR / 'training_history.csv'), matrix, mistakes)
    return metrics


def markdown_table(frame):
    rows = [list(frame.columns)] + frame.astype(str).values.tolist()
    rows.insert(1, ['---'] * len(frame.columns))
    return '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)


def write_results_report():
    metrics = json.loads((RESULT_DIR / 'metrics.json').read_text(encoding='utf-8'))
    history = pd.read_csv(RESULT_DIR / 'training_history.csv')
    report = pd.read_csv(RESULT_DIR / 'classification_report.csv', index_col=0)
    matrix = pd.read_csv(RESULT_DIR / 'confusion_matrix.csv', index_col=0)
    comparison = pd.read_csv(RESULT_DIR / 'experiment_comparison.csv')
    best = int(history['val_loss'].argmin())
    best_row = history.iloc[best]
    per_class = report.loc[config.CLASS_NAMES].round(4).reset_index()
    per_class['support'] = per_class['support'].astype(int)
    errors = [(int(matrix.loc[a, b]), a, b) for a in config.CLASS_NAMES
              for b in config.CLASS_NAMES if a != b]
    errors = pd.DataFrame(sorted(errors, reverse=True)[:4],
                          columns=['Số ảnh', 'Nhãn thật', 'Dự đoán'])
    text = f'''# Kết quả chạy Basic Neural Network

Phương pháp và cách chạy: [basic_nn_method.md](basic_nn_method.md).
Số liệu dưới đây được lấy từ CSV và `metrics.json` của checkpoint đang lưu.

## 1. Kết quả test

- Checkpoint: `models/basic_nn_best.keras`; {metrics['parameter_count']:,} tham số.
- Số ảnh test: {int(report.loc['macro avg', 'support'])}.
- Accuracy: **{metrics['test_accuracy']:.3%}**.
- Macro F1: **{metrics['macro_f1']:.4f}**; weighted F1: **{metrics['weighted_f1']:.4f}**.
- Test loss: **{metrics['test_loss']:.6f}**; số ảnh sai: **{metrics['misclassified_count']}**.

## 2. Huấn luyện

Đã chạy {len(history)} epochs, checkpoint tốt nhất tại epoch **{best + 1}** theo validation loss.
Tại epoch đó: train accuracy {best_row.accuracy:.3%}, train loss {best_row.loss:.6f},
validation accuracy {best_row.val_accuracy:.3%}, validation loss {best_row.val_loss:.6f}.
Cấu hình thực tế của lần chạy được lưu trong `metrics.json`:

```json
{json.dumps(metrics['training_config'], indent=2)}
```

![Accuracy và loss](accuracy_loss_curves.png)

Train có augmentation và Dropout, val không có; train accuracy thấp hơn val
chưa đủ để kết luận underfitting hoặc overfitting. Cần xem cả loss và đánh giá
train trong chế độ inference trước khi điều chỉnh regularization.

## 3. Từng lớp và confusion matrix

{markdown_table(per_class)}

Hàng là nhãn thật, cột là dự đoán:

{markdown_table(matrix.reset_index())}

![Confusion matrix](confusion_matrix.png)

## 4. Phân tích lỗi

{markdown_table(errors)}

`fire → nofire`: {int(matrix.loc['fire', 'nofire'])} ảnh;
`smokefire → nofire`: {int(matrix.loc['smokefire', 'nofire'])} ảnh.
F1 thấp nhất ở lớp **{report.loc[config.CLASS_NAMES, 'f1-score'].idxmin()}**.
Flatten không khai thác trực tiếp quan hệ không gian giữa các pixel như CNN.

![Ví dụ sai](misclassified_samples.png)

Ảnh minh họa là tối đa 16 lỗi đầu tiên theo thứ tự manifest.
Toàn bộ lỗi và confidence nằm trong `misclassified.csv`.

## 5. So sánh lịch sử

{markdown_table(comparison.round(6))}

Bảng này ghi các lần chạy trước khi dọn phiên bản, không tự cập nhật theo lần
train mới. Augmentation đạt test accuracy và Macro F1 cao nhất trong lịch sử;
bản Adam 0.0005 không augmentation có validation loss thấp hơn.
Việc chọn cũ dựa trên test là hồi cứu; không dùng bảng này để tiếp tục tuning.
Các lần cải thiện tiếp theo phải chọn bằng validation, test chỉ dùng báo cáo cuối.
Cần holdout mới để đánh giá độc lập sau lựa chọn lịch sử.

## 6. Kiểm chứng và tệp nguồn

Checkpoint đã được đánh giá lại trên test. Các lần thử huấn luyện và mức
thay đổi accuracy được ghi ở mục tuning bên dưới. Metrics, history, classification
report, confusion matrix và danh sách lỗi là các tệp nguồn trong thư mục này.
Giữ một checkpoint, một file code Basic NN và hai tài liệu phương pháp/kết quả.
'''
    tuning_path = RESULT_DIR / 'tuning_summary.json'
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text(encoding='utf-8'))
        text += f'''
## 7. Thử cải thiện cấu hình

Chọn theo validation loss nhỏ nhất, không dùng test để chọn các thử nghiệm.
Các thử nghiệm chỉ thay một yếu tố, giữ seed, split, augmentation và kiến trúc.

{markdown_table(pd.DataFrame(tuning['trials']).round(6))}

Cấu hình được chọn trong lần thử: **{tuning['selected']['trial']}**.
Test accuracy trước thử: **{tuning['previous_test_accuracy']:.3%}**;
sau chọn bằng validation: **{tuning['selected_test_accuracy']:.3%}**.
Chênh lệch: **{tuning['change_percentage_points']:+.3f} điểm phần trăm**.
Mốc lịch sử trước lần train lại là {tuning['historical_test_accuracy']:.3%}.
Số liệu này thuộc lần tuning đã lưu; các lần train sau có thể khác.
Một seed chưa đủ để kết luận cải thiện ổn định qua nhiều lần chạy.
'''
    (RESULT_DIR / 'basic_nn_results.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--evaluate-only', action='store_true')
    modes.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        write_results_report()
        print('Results report updated.')
        return
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    train, val, test = (frames[split] for split in config.SPLITS)
    print(f'Train: {len(train)}, validation: {len(val)}, test: {len(test)}')
    settings = dict(input_size=list(INPUT_SIZE), dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
        epochs=EPOCHS, patience=PATIENCE, seed=config.RANDOM_SEED,
        reduce_lr_on_plateau=True, data_pipeline='src.datasets (shared augmentation)')
    if args.evaluate_only:
        if not MODEL_PATH.exists() or not (RESULT_DIR / 'training_history.csv').exists():
            raise FileNotFoundError('Missing saved model or training history')
        if (RESULT_DIR / 'metrics.json').exists():
            settings = json.loads((RESULT_DIR / 'metrics.json').read_text())['training_config']
        if tuple(settings['input_size']) != INPUT_SIZE:
            raise ValueError('INPUT_SIZE must match the saved checkpoint configuration')
    else:
        model = build_model()
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
            tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE),
        ]
        history = model.fit(make_dataset(train, training=True, batch_size=BATCH_SIZE),
                            validation_data=make_dataset(val, batch_size=BATCH_SIZE),
                            epochs=EPOCHS, callbacks=callbacks, verbose=1)
        pd.DataFrame(history.history).to_csv(RESULT_DIR / 'training_history.csv', index=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    if not any(isinstance(layer, tf.keras.layers.Rescaling) for layer in model.layers):
        raise ValueError('Checkpoint predates the shared data pipeline (expects 64x64 inputs '
                         'in [0, 1]); retrain with python -m src.models.basic_nn')
    metrics = evaluate(model, test, settings)
    write_results_report()
    print(f"Done. Test accuracy: {metrics['test_accuracy']:.3%}; Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
