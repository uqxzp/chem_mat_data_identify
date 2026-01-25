import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve as sklearn_roc_curve


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line.strip())


def compute_roc_metrics(scores: np.ndarray, labels: np.ndarray):
    fpr, sensitivity, thresholds = sklearn_roc_curve(labels, scores)
    return fpr, sensitivity, thresholds


def calculate_operating_point(scores: np.ndarray, labels: np.ndarray, threshold: float):
    predictions = scores >= threshold

    true_positives = np.sum(predictions & (labels == 1))
    false_positives = np.sum(predictions & (labels == 0))
    false_negatives = np.sum(~predictions & (labels == 1))
    true_negatives = np.sum(~predictions & (labels == 0))

    true_positive_rate = true_positives / (true_positives + false_negatives)
    false_positive_rate = false_positives / (false_positives + true_negatives)

    return false_positive_rate, true_positive_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.05, 0.075, 0.1])
    args = parser.parse_args()

    scores_list = []
    labels_list = []
    for record in iter_jsonl(args.jsonl):
        scores_list.append(record["prediction"])
        labels_list.append(record["label"])

    scores = np.array(scores_list, dtype=float)
    labels = np.array(labels_list, dtype=int)

    fpr, tpr, _ = compute_roc_metrics(scores, labels)
    area_under_curve = float(sklearn_auc(fpr, tpr))

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={area_under_curve:.4f})", color="black")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    for threshold in args.thresholds:
        fpr_t, sens_t = calculate_operating_point(scores, labels, threshold)
        plt.scatter([fpr_t], [sens_t], label=f"Threshold {threshold:g}")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
