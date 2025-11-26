import argparse
import json
from pathlib import Path

# Prints all titles in the given jsonl file. By default, it prints all flagged titles that aren't marked as positive yet.
# PYTHONPATH=. python data/scripts/print_titles.py 


def load_titles(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    titles = [r["title"] for r in records]
    return titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("data/production/scored_flagged.jsonl"))
    parser.add_argument("--dedup", type=Path, default=Path("data/positives.jsonl"))
    parser.add_argument("--batch_size", type=int, default=0)
    args = parser.parse_args()

    to_print = load_titles(args.jsonl)
    if args.jsonl != args.dedup:
        dedup = load_titles(args.dedup)
        to_print = [t for t in to_print if t not in dedup]

    for i, t in enumerate(to_print, start=1):
        print(f"{i}: {t}")
        if args.batch_size > 1 and i % args.batch_size == 0:
            print()


if __name__ == "__main__":
    main()
