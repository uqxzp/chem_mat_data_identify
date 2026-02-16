import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap


def visualize_predictions(path: str) -> None:
    plt.rcParams.update({"font.size": 16})
    fig = plt.figure(figsize=(8, 6))
    df = pd.read_json(path, lines=True, dtype={"label": "int64", "pred": "float64"})
    pred_ys = df["prediction"].to_numpy()
    true_ys = df.get("label", pd.Series(-1, index=df.index)).to_numpy()

    cmap = ListedColormap(["red", "green"])
    plt.scatter(
        range(len(true_ys)),
        pred_ys,
        c=true_ys,
        cmap=cmap,
        vmin=0,
        vmax=1,
        edgecolor="k",
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Score")
    plt.title("Predictions vs. True Labels")
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Dataset",
            markerfacecolor="green",
            markersize=8,
            markeredgecolor="k",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="No Dataset",
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
    fig.savefig("data/predictions.png", dpi=300, bbox_inches="tight")
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
