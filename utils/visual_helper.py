"""Helper for visualizing the predictions from the SLM classifier."""
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def visualize_predictions(path: str) -> None:
    plt.rcParams.update({"font.size": 16})
    fig = plt.figure(figsize=(8, 6))
    df = pd.read_json(path, lines=True, dtype={"prediction": "float64"})
    pred_ys = df["prediction"].to_numpy()

    plt.scatter(
        range(len(pred_ys)),
        pred_ys,
        color="steelblue",
        edgecolor="k",
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Score")
    plt.title("Predicted Scores")
    plt.tight_layout()
    fig.savefig("data/predictions.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/scored_all.jsonl")
    args = parser.parse_args()

    visualize_predictions(args.jsonl)


if __name__ == "__main__":
    main()
