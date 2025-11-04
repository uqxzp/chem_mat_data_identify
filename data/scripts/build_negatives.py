import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from utils.fetch_openalex import (norm_str, search_hard_negatives_chem_and_mat,
                                  search_random_negatives_chem_and_mat)

# Example usage: PYTHONPATH=. python data/scripts/build_negatives.py   --positives_jsonl data/positives.jsonl   --outfile data/negatives.jsonl

YEAR_LOWER_BOUND = 2010

# terms indicating hard negatives
HARD_NEGATIVE_TERMS = [
    "dataset",
    "data set",
    "database",
    "benchmark",
    "corpus",
    "collection",
    "screening",
    "high-throughput",
    "high throughput",
    "hte",
    "combinatorial",
    "large-scale",
    "library",
    "big data",
    "data mining",
]

# phrases implying real data release; exclude from negatives
RELEASE_HINTS = [
    "we release",
    "we present a dataset",
    "we introduce a dataset",
    "we publish a dataset",
    "we make.*dataset.*available",
    "available at",
    "available on",
    "github.com",
    "gitlab.com",
    "figshare",
    "zenodo",
    "osf.io",
    "data repository",
    "supplementary dataset",
    "downloadable dataset",
    "data record",
    "dataset.*doi",
    "https://doi.org/",
]
RE_RELEASE = re.compile("|".join(RELEASE_HINTS), flags=re.IGNORECASE)


def load_positive_titles(p: Path) -> set[str]:
    titles: set[str] = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                title = json.loads(line).get("title")
            except json.JSONDecodeError:
                continue
            t = norm_str(title)
            if t:
                titles.add(t)
    return titles


def likely_release(text: str) -> bool:
    return bool(text and RE_RELEASE.search(text))


def to_negative_record(rec: dict, hard: int) -> dict:
    return {
        "title": rec.get("title"),
        "abstract": rec.get("abstract"),
        "label": 0,
        "hard_negative": hard,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives_jsonl", type=Path, required=True)
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--random_n", type=int, default=250)
    ap.add_argument("--hard_n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    positive_titles = load_positive_titles(args.positives_jsonl)

    seen_titles: set[str] = set()

    # random negatives
    random_negatives: list[dict] = []
    random_candidates = search_random_negatives_chem_and_mat(
        sample_size=args.random_n * 2,  # oversample for filtering
        seed=args.seed,
        year_lower_bound=YEAR_LOWER_BOUND,
    )

    for rec in random_candidates:
        title = rec.get("title")
        normalized_title = norm_str(title)

        if not title:
            continue
        if normalized_title in positive_titles or normalized_title in seen_titles:
            continue
        if likely_release(rec.get("abstract")):
            continue
        random_negatives.append(to_negative_record(rec, 0))
        seen_titles.add(normalized_title)

        if len(random_negatives) >= args.random_n:
            break

    # hard negatives
    hard_negatives: list[dict] = []
    hard_candidates = search_hard_negatives_chem_and_mat(
        HARD_NEGATIVE_TERMS,
        limit=args.hard_n * 2,  # oversample for filtering
        year_lower_bound=YEAR_LOWER_BOUND,
    )

    for rec in hard_candidates:
        title = rec.get("title")
        normalized_title = norm_str(title)

        if not title:
            continue
        if normalized_title in positive_titles or normalized_title in seen_titles:
            continue
        if likely_release(rec.get("abstract")):
            continue
        hard_negatives.append(to_negative_record(rec, 1))
        seen_titles.add(normalized_title)

        if len(hard_negatives) >= args.hard_n:
            break

    # write
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with args.outfile.open("w", encoding="utf-8") as out:
        for line in random_negatives + hard_negatives:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Random: {len(random_negatives)}, Hard: {len(hard_negatives)}")


if __name__ == "__main__":
    main()
