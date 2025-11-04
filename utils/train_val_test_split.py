import json
import random
from pathlib import Path

POS_PATH = Path("data/positives.jsonl")
NEG_PATH = Path("data/negatives.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


def create_train_val_test_split(
    n_pos_val: int, n_neg_val: int, n_pos_test: int, n_neg_test: int
):
    if not POS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {POS_PATH}")
    if not NEG_PATH.exists():
        raise FileNotFoundError(f"Missing file: {NEG_PATH}")

    positives = read_jsonl(POS_PATH)
    negatives = read_jsonl(NEG_PATH)

    if (
        n_pos_val < 0
        or n_neg_val < 0
        or n_pos_test < 0
        or n_neg_test < 0
        or n_pos_val + n_pos_test > len(positives)
        or n_neg_val + n_neg_test > len(negatives)
    ):
        raise ValueError("Invalid arguments")

    random.shuffle(positives)
    random.shuffle(negatives)

    test_pos = positives[:n_pos_test]
    test_neg = negatives[:n_neg_test]
    test = test_pos + test_neg

    val_pos = positives[n_pos_test : n_pos_test + n_pos_val]
    val_neg = negatives[n_neg_test : n_neg_test + n_neg_val]
    val = val_pos + val_neg

    train_pos = positives[n_pos_test + n_pos_val :]
    train_neg = negatives[n_neg_test + n_neg_val :]
    train = train_pos + train_neg

    random.shuffle(test)
    random.shuffle(val)
    random.shuffle(train)

    return train, val, test
