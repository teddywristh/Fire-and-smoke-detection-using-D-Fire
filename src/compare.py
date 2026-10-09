"""Paired bootstrap of macro F1 for every run in results/leaderboard.csv.

    python -m src.compare [reference_run]

Resamples the val and test images 10,000 times, scores every run on the same
resamples, and writes results/comparison.csv with each run's macro F1, its
95% interval, and the difference from the reference run (default:
frozen_resnet50) with its 95% interval.
"""

import sys

import numpy as np
import pandas as pd

import config


def macro_f1(truth, predicted):
    """Macro F1 for each row of (resamples, images) label arrays."""
    scores = []
    for c in range(len(config.CLASS_NAMES)):
        tp = ((truth == c) & (predicted == c)).sum(-1)
        wrong = ((truth == c) != (predicted == c)).sum(-1)
        scores.append(np.where(2 * tp + wrong > 0, 2 * tp / np.maximum(2 * tp + wrong, 1), 0.0))
    return np.mean(scores, 0)


def compare(reference="frozen_resnet50", resamples=10_000, seed=0):
    runs = pd.read_csv(config.RESULTS_DIR / "leaderboard.csv").run
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("val", "test"):
        predictions = {run: pd.read_csv(config.RESULTS_DIR / run / f"{split}_predictions.csv") for run in runs}
        truth = predictions[reference].label.map(config.CLASS_TO_INDEX).to_numpy()
        index = rng.integers(0, len(truth), size=(resamples, len(truth)))
        labels = {run: frame.predicted.map(config.CLASS_TO_INDEX).to_numpy() for run, frame in predictions.items()}
        reference_boot = macro_f1(truth[index], labels[reference][index])
        for run, predicted in labels.items():
            assert predictions[run].filepath.equals(predictions[reference].filepath)
            boot = macro_f1(truth[index], predicted[index])
            delta = boot - reference_boot
            rows.append({
                "split": split, "run": run,
                "macro_f1": macro_f1(truth, predicted), "f1_low": np.percentile(boot, 2.5),
                "f1_high": np.percentile(boot, 97.5),
                "delta": macro_f1(truth, predicted) - macro_f1(truth, labels[reference]),
                "delta_low": np.percentile(delta, 2.5), "delta_high": np.percentile(delta, 97.5),
            })
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "comparison.csv", index=False, float_format="%.4f")
    return table


if __name__ == "__main__":
    print(compare(*sys.argv[1:]).round(4).to_string(index=False))
