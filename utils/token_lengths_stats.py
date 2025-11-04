import argparse
import json
import statistics

from transformers import AutoTokenizer

# Example usage: PYTHONPATH=. python utils/token_lengths_stats.py --jsonl data/negatives.jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    lengths = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            ids = tok(
                ex.get("title") or "",
                text_pair=ex.get("abstract") or "",
                truncation=False,
            )["input_ids"]
            lengths.append(len(ids))

    if not lengths:
        stats = {"count": 0, "min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    else:
        data = sorted(lengths)
        q = statistics.quantiles(data, n=100, method="inclusive")
        stats = {
            "count": len(data),
            "min": data[0],
            "p50": int(q[49]),
            "p90": int(q[89]),
            "max": data[-1],
            "mean": sum(data) / len(data),
        }

    print(json.dumps({"title_plus_abstract_tokens": stats}, indent=2))


if __name__ == "__main__":
    main()
