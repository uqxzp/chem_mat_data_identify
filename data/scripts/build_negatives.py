import argparse, json, os, re, sys, unicodedata
from pathlib import Path
from typing import Dict, List, Set

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from data.scripts.fetch_openalex import (
    iter_random_works_in_chem_and_mat,
    search_works_hard_negative_candidates,
)

# terms indicating hard negatives
HARD_NEGATIVE_TERMS = [
    "dataset", "data set", "database", "benchmark", "corpus", "collection",
    "screening", "high-throughput", "high throughput", "hte", "combinatorial",
    "large-scale", "library", "big data", "data mining",
]

# phrases implying real data release; exclude from negatives
RELEASE_HINTS = [
    "we release", "we present a dataset", "we introduce a dataset",
    "we publish a dataset", "we make.*dataset.*available",
    "data available at", "available at", "available on",
    "github.com", "gitlab.com", "figshare", "zenodo", "osf.io",
    "data repository", "supplementary dataset", "downloadable dataset",
    "data record", "dataset.*doi", "https://doi.org/",
]
RE_RELEASE = re.compile("|".join(RELEASE_HINTS), flags=re.IGNORECASE)

def _norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").strip().lower()
    return re.sub(r"\s+", " ", t)

def _load_positive_titles(p: Path) -> Set[str]:
    if not p.exists():
        return set()
    titles: Set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            titles.add(_norm_title(obj.get("title", "")))
    return titles

def _looks_like_release(text: str) -> bool:
    return bool(text and RE_RELEASE.search(text))

def _minimal_record(rec: Dict, hard: bool) -> Dict:
    return {
        "title": rec.get("title", "") or "",
        "abstract": rec.get("abstract", "") or "",
        "label": 0,
        "hard_negative": hard,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives_jsonl", type=Path, required=True)  # for title de-dupe
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--mailto", type=str, default=os.environ.get("OPENALEX_MAILTO"))
    ap.add_argument("--random_n", type=int, default=200)
    ap.add_argument("--hard_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--year_from", type=int, default=2000)
    args = ap.parse_args()

    pos_titles = _load_positive_titles(args.positives_jsonl)

    # random negatives
    random_negs: List[Dict] = []
    for rec in iter_random_works_in_chem_and_mat(
        sample_size=args.random_n * 2,  # small oversample for filtering
        seed=args.seed,
        year_lower_bound=args.year_from,
        mailto=args.mailto,
    ):
        title = rec.get("title")
        if not title:
            continue
        if _norm_title(title) in pos_titles:
            continue
        text = f'{rec.get("title","")} {rec.get("abstract","")}'
        if _looks_like_release(text):
            continue
        random_negs.append(_minimal_record(rec, hard=False))
        if len(random_negs) >= args.random_n:
            break

    # hard negatives
    hard_candidates = search_works_hard_negative_candidates(
        HARD_NEGATIVE_TERMS,
        limit=args.hard_n * 3,  # small oversample for filtering
        mailto=args.mailto,
        year_lower_bound=args.year_from,
    )
    hard_negs: List[Dict] = []
    seen = { _norm_title(x["title"]) for x in random_negs }
    for rec in hard_candidates:
        title = rec.get("title")
        if not title:
            continue
        nt = _norm_title(title)
        if nt in pos_titles or nt in seen:
            continue
        text = f'{rec.get("title","")} {rec.get("abstract","")}'
        if _looks_like_release(text):
            continue
        hard_negs.append(_minimal_record(rec, hard=True))
        seen.add(nt)
        if len(hard_negs) >= args.hard_n:
            break

    # write
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with args.outfile.open("w", encoding="utf-8") as out:
        for ex in random_negs + hard_negs:
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"{len(random_negs)+len(hard_negs)} negatives → {args.outfile}")
    print(f"  random={len(random_negs)}  hard={len(hard_negs)}")

if __name__ == "__main__":
    main()
