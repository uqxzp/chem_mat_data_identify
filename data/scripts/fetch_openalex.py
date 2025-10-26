from __future__ import annotations
import time, requests
from typing import Optional, Dict, List, Iterator

# helper class for building negatives dataset

OA_BASE = "https://api.openalex.org"

def _reconstruct_from_inverted_index(inv_idx: Dict[str, List[int]]) -> str:
    # rebuild abstract text from openalex index
    if not inv_idx:
        return ""
    max_pos = 0
    for positions in inv_idx.values():
        if positions:
            max_pos = max(max_pos, max(positions) + 1)
    words = [""] * max_pos
    for token, positions in inv_idx.items():
        for p in positions:
            if 0 <= p < max_pos:
                words[p] = token
    return " ".join(w for w in words if w)

def _to_minimal_record(obj: Dict) -> Dict[str, str]:
    # only essential columns
    title = obj.get("title") or ""
    inv_idx = obj.get("abstract_inverted_index") or None
    abstract = obj.get("abstract") or ""
    abstract_text = _reconstruct_from_inverted_index(inv_idx) if inv_idx else abstract
    return {
        "title": title,
        "abstract": abstract_text,
        "openalex_id": obj.get("id", ""),
    }

def _session(mailto: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": f"dataset-miner/0.1 (+{mailto})" if mailto else "dataset-miner/0.1"})
    if mailto:
        s.params = {"mailto": mailto}
    return s

def iter_random_works_in_chem_and_mat(
    sample_size: int,
    seed: Optional[int] = None,
    year_lower_bound: int = 2000,
    include_types: str = "journal-article|proceedings-article|book-chapter|report|review-article",
    mailto: Optional[str] = None,
    per_page: int = 200,
    sleep_s: float = 0.0,
) -> Iterator[Dict[str, str]]:
    # random sample from chemistry (fields/16) and materials (fields/25) with abstracts
    filters = [
        "has_abstract:true",
        f"publication_year:>{year_lower_bound-1}",
        "primary_topic.field.id:fields/16|fields/25",
        f"type:{include_types}",
    ]
    params = {
        "sample": sample_size,
        "per-page": per_page,
        "filter": ",".join(filters),
        "select": "id,title,abstract_inverted_index",
    }
    if seed is not None:
        params["seed"] = str(seed)
    with _session(mailto) as s:
        r = s.get(f"{OA_BASE}/works", params=params, timeout=30)
        r.raise_for_status()
        for obj in r.json().get("results", []):
            yield _to_minimal_record(obj)
            if sleep_s:
                time.sleep(sleep_s)

def search_works_hard_negative_candidates(
    query_terms: List[str],
    limit: int = 500,
    mailto: Optional[str] = None,
    year_lower_bound: int = 2000,
    include_types: str = "journal-article|proceedings-article|book-chapter|report|review-article",
) -> List[Dict[str, str]]:
    # keyword search likely to surface dataset-like texts (post-filter later)
    results: List[Dict[str, str]] = []
    filters = [
        "has_abstract:true",
        f"publication_year:>{year_lower_bound-1}",
        "primary_topic.field.id:fields/16|fields/25",
        f"type:{include_types}",
    ]
    with _session(mailto) as s:
        for term in query_terms:
            if len(results) >= limit:
                break
            params = {
                "search": term,
                "per-page": 200,
                "filter": ",".join(filters),
                "select": "id,title,abstract_inverted_index",
                "sort": "relevance_score:desc",
            }
            page = 1
            while len(results) < limit:
                params["page"] = page
                r = s.get(f"{OA_BASE}/works", params=params, timeout=30)
                if r.status_code != 200:
                    break
                chunk = r.json().get("results", [])
                if not chunk:
                    break
                for obj in chunk:
                    results.append(_to_minimal_record(obj))
                    if len(results) >= limit:
                        break
                page += 1
    # de-dup by openalex id
    seen = set()
    uniq: List[Dict[str, str]] = []
    for rec in results:
        oid = rec.get("openalex_id")
        if oid and oid not in seen:
            uniq.append(rec)
            seen.add(oid)
    return uniq[:limit]
