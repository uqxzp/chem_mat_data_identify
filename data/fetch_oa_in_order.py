import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from utils.fetch_openalex import fetch, reconstruct_abstract

DEFAULT_FILTERS = [
    "has_abstract:true",
    "concepts.id:https://openalex.org/C185592680|https://openalex.org/C192562407",
    "type:journal-article|proceedings-article|book-chapter|report|review-article",
]

SELECT_FIELDS = "id,title,abstract_inverted_index,publication_year"

CURSOR: Path = Path("data/unlabeled_openalex.cursor")
SEEN_IDS: Path = Path("data/unlabeled_openalex.ids")

def load_seen_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def write_cursor(path: Path, cursor: str | None):
    if cursor is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cursor, encoding="utf-8")


def make_filters(year_lower_bound: int) -> str:
    filters = list(DEFAULT_FILTERS)
    filters.append(f"publication_year:>{year_lower_bound-1}")
    return ",".join(filters)


def to_record(obj: dict) -> dict:
    return {
        "openalex_id": obj.get("id"),
        "title": obj.get("title"),
        "abstract": reconstruct_abstract(obj.get("abstract_inverted_index")),
        "publication_year": obj.get("publication_year"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", type=Path, default=Path("data/unlabeled_openalex.jsonl"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--year_lower_bound", type=int, default=2010)
    args = parser.parse_args()

    if args.limit <= 0:
        print("Invalid limit")
        return

    filters = make_filters(args.year_lower_bound)
    seen_ids = load_seen_ids(SEEN_IDS)
    cursor = CURSOR.read_text(encoding="utf-8").strip() if CURSOR.exists() else "*"
    fetched = 0
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    SEEN_IDS.parent.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=args.limit, desc="Fetching works", unit="record")

    with (
        args.outfile.open("a", encoding="utf-8") as out_f,
        SEEN_IDS.open("a", encoding="utf-8") as seen_f,
    ):
        while fetched < args.limit:
            params = {
                "filter": filters,
                "select": SELECT_FIELDS,
                "per-page": min(200, args.limit - fetched),
                "cursor": cursor,
                "sort": "publication_year:desc,publication_date:desc",
            }
            response = fetch("/works", params=params)
            if response is None or response.status_code != 200:
                print("OpenAlex request failed; stopping early.")
                break

            payload = response.json()
            items = payload.get("results") or []
            if not items:
                print("No more results returned; stopping.")
                break

            new_cursor = payload.get("meta", {}).get("next_cursor")
            if not new_cursor:
                print("Missing next_cursor; stopping to avoid duplicate traversal.")
                break

            for obj in items:
                oid = obj.get("id")
                if not oid or oid in seen_ids:
                    continue
                record = to_record(obj)
                if not record["title"] or not record["abstract"]:
                    continue
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_f.write(oid + "\n")
                seen_f.flush()
                seen_ids.add(oid)
                fetched += 1
                progress.update(1)
                if fetched >= args.limit:
                    break

            cursor = new_cursor
            write_cursor(CURSOR, cursor)

    progress.close()
    print(f"Wrote {fetched} new records to {args.outfile}.")


if __name__ == "__main__":
    main()
