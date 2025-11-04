from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from random import Random
from typing import Iterator

from requests import RequestException, Response, Session
from requests.adapters import HTTPAdapter
from urllib3 import Retry

OA_BASE = "https://api.openalex.org"


def session() -> Session:
    s = Session()
    s.headers.update({"User-Agent": "chem-mat-detector/0.1"})
    return s


def fetch(
    path: str, params: dict, tries: int = 3, timeout: int = 10
) -> Response | None:
    retries = Retry(
        total=tries - 1,  # attempts = total + 1
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    with Session() as s:
        s.mount("https://", HTTPAdapter(max_retries=retries))

        try:
            url = f"{OA_BASE}{path}"
            res = s.get(url, params=params, timeout=timeout)
        except RequestException:
            return None

        return res if res.ok else None


def reconstruct_abstract(inv_idx: dict | None) -> str:
    if not inv_idx:
        return ""
    pos = [p for ps in inv_idx.values() for p in ps]
    if not pos:
        return ""
    words = [""] * (max(pos) + 1)
    for tok, ps in inv_idx.items():
        for p in ps:
            words[p] = tok
    return " ".join(filter(None, words))


def norm_str(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    return " ".join(s.split())


def best_title_match(query: str, items: list[dict]) -> dict | None:
    q = norm_str(query)
    m = SequenceMatcher(a=q, autojunk=False)
    best_match, best_score = None, -1
    for obj in items:
        title = norm_str(obj.get("title") or obj.get("display_name") or "")
        if not title:
            continue
        if title == q:
            return obj  # perfect match
        m.set_seq2(title)
        score = m.ratio()
        if score > best_score:
            best_match, best_score = obj, score
    return best_match


def to_min_record(obj: dict) -> dict[str, str]:
    return {
        "title": obj.get("title"),
        "abstract": reconstruct_abstract(obj.get("abstract_inverted_index")),
    }


def search_random_negatives_chem_and_mat(
    sample_size: int, seed: int, year_lower_bound: int
) -> Iterator[dict[str, str]]:
    if sample_size <= 0:
        return

    filters = [
        "has_abstract:true",
        f"publication_year:>{year_lower_bound-1}",
        "primary_topic.field.id:fields/16|fields/25",
        "type:journal-article|proceedings-article|book-chapter|report|review-article",
    ]

    max_batch = 200  # OpenAlex per-request upper bound
    rndm = Random(seed)
    requested = 0
    seen_ids: set[str] = set()
    attempts = 0

    while requested < sample_size:
        batch_size = min(max_batch, sample_size - requested)
        params = {
            "sample": batch_size,
            "filter": ",".join(filters),
            "select": "id,title,abstract_inverted_index",
        }
        params["seed"] = str(rndm.getrandbits(31))

        r = fetch("/works", params)
        if not r or r.status_code != 200:
            continue

        added_this_round = 0
        for obj in r.json().get("results", ()):
            oid = obj.get("id")
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)
            requested += 1
            added_this_round += 1
            yield to_min_record(obj)
            if requested >= sample_size:
                return


def search_hard_negatives_chem_and_mat(
    query_terms: list[str], limit: int = 500, year_lower_bound: int = 2000
) -> list[dict[str, str]]:
    filter_str = ",".join(
        [
            "has_abstract:true",
            f"publication_year:>{year_lower_bound-1}",
            "primary_topic.field.id:fields/16|fields/25",
            "type:journal-article|proceedings-article|book-chapter|report|review-article",
        ]
    )

    seen: set[str] = set()
    results: list[dict[str, str]] = []

    for term in query_terms:
        if len(results) >= limit:
            break

        params = {
            "search": term,
            "filter": filter_str,
            "select": "id,title,abstract_inverted_index",
            "page": 1,
        }

        while len(results) < limit:
            r = fetch("/works", params)
            if not r or r.status_code != 200:
                break

            for obj in r.json().get("results", ()):
                oid = obj.get("id")
                if not oid or oid in seen:
                    continue
                seen.add(oid)
                results.append(to_min_record(obj))
                if len(results) >= limit:
                    break

            params["page"] += 1

    return results


def fetch_by_title(title: str) -> dict | None:
    # try exact first, then broad
    for params in ({"filter": f"title.search:{title}"}, {"search": title}):
        r = fetch(
            "/works",
            {**params, "select": "id,title,display_name,abstract_inverted_index"},
        )
        items = (r.json().get("results") or ()) if (r and r.status_code == 200) else ()
        if items:
            break
    else:
        return None

    obj = best_title_match(title, items) or items[0]
    return {
        "title": obj.get("title") or obj.get("display_name") or "",
        "abstract": reconstruct_abstract(obj.get("abstract_inverted_index")),
    }
