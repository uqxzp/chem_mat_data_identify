"""Helper for loading train/validation splits and the test set for model development and evaluation."""
import json
import random
from pathlib import Path

POS_PATH = Path("data/positives.jsonl")
NEG_PATH = Path("data/negatives.jsonl")
TEST_PATH = Path("data/test_set.jsonl")


def read_jsonl(path: Path) -> list[dict]: 
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_train_val_test(
    n_pos_val: int, n_neg_val: int
):
    positives = read_jsonl(POS_PATH)
    negatives = read_jsonl(NEG_PATH)
    test = read_jsonl(TEST_PATH)

    if (
        n_pos_val < 0
        or n_neg_val < 0
        or n_pos_val > len(positives)
        or n_neg_val > len(negatives)
    ):
        raise ValueError("Invalid arguments")

    random.shuffle(positives)
    random.shuffle(negatives)

    val_pos = positives[:n_pos_val]
    val_neg = negatives[:n_neg_val]
    val = val_pos + val_neg

    train_pos = positives[n_pos_val:]
    train_neg = negatives[n_neg_val:]
    train = train_pos + train_neg
    
    random.shuffle(val)
    random.shuffle(train)

    return train, val, test
