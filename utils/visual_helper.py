import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def visualize_predictions(path: str) -> None:
    df = pd.read_json(path, lines=True, dtype={"label": "int64", "pred": "float64"})
    pred_ys = df["prediction"].to_numpy()
    true_ys = df["label"].to_numpy()

    plt.scatter(
        range(len(true_ys)),
        pred_ys,
        c=true_ys,
        cmap="bwr",
        edgecolor="k",
    )
    plt.xlabel("Sample index")
    plt.ylabel("Predicted probability")
    plt.title("Predictions vs. true labels")
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Negative",
            markerfacecolor="blue",
            markersize=8,
            markeredgecolor="k",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Positive",
            markerfacecolor="red",
            markersize=8,
            markeredgecolor="k",
        ),
    ]
    plt.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl", default="outputs/classifier_448/test_predictions.jsonl"
    )
    args = parser.parse_args()

    visualize_predictions(args.jsonl)


if __name__ == "__main__":
    main()
