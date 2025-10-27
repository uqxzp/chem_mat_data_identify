from __future__ import annotations
import time, re, requests
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Iterator

OA_BASE = "https://api.openalex.org"

# ---- HTTP (session + retry) ----
def _session(mailto: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    ua = "chem-mat-detector/0.1"
    s.headers.update({"User-Agent": f"{ua} (+{mailto})" if mailto else ua})
    if mailto:
        s.params = {"mailto": mailto}
    return s

def _get(path_or_url: str, params: Dict, tries: int = 4, timeout: int = 30, mailto: Optional[str] = None) -> Optional[requests.Response]:
    # path_or_url: full URL or '/works...'
    backoff = 1.0
    url = path_or_url if path_or_url.startswith("http") else f"{OA_BASE}{path_or_url}"
    with _session(mailto) as s:
        for _ in range(tries):
            try:
                r = s.get(url, params=params, timeout=timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff); backoff *= 2; continue
                return r
            except requests.RequestException:
                time.sleep(backoff); backoff *= 2
    return None

# ---- text helpers ----
def reconstruct_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> str:
    if not inv_idx: return ""
    max_pos = max((max(v) for v in inv_idx.values() if v), default=-1) + 1
    words = [""] * max_pos
    for tok, poss in inv_idx.items():
        for p in poss:
            if 0 <= p < max_pos:
                words[p] = tok
    return " ".join(w for w in words if w)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _best_title_match(query: str, items: List[Dict]) -> Optional[Dict]:
    q = _norm(query); best, best_score = None, -1.0
    for obj in items:
        t = _norm(obj.get("title") or obj.get("display_name") or "")
        sc = SequenceMatcher(None, q, t).ratio()
        if sc > best_score:
            best, best_score = obj, sc
    return best

def _to_minimal_record(obj: Dict) -> Dict[str, str]:
    return {
        "title": obj.get("title") or "",
        "abstract": reconstruct_abstract(obj.get("abstract_inverted_index")),
        "openalex_id": obj.get("id", ""),
    }

# ---- negatives dataset helpers ----
def iter_random_works_in_chem_and_mat(
    sample_size: int,
    seed: Optional[int] = None,
    year_lower_bound: int = 2000,
    include_types: str = "journal-article|proceedings-article|book-chapter|report|review-article",
    mailto: Optional[str] = None,
    per_page: int = 200,
    sleep_s: float = 0.0,
) -> Iterator[Dict[str, str]]:
    # random sample from chemistry (fields/16) and materials (fields/25)
    filters = [
        "has_abstract:true",
        f"publication_year:>{year_lower_bound-1}",
        "primary_topic.field.id:fields/16|fields/25",
        f"type:{include_types}",
    ]
    params = {
        "sample": sample_size,
        "per_page": per_page,
        "filter": ",".join(filters),
        "select": "id,title,abstract_inverted_index",
    }
    if seed is not None:
        params["seed"] = str(seed)
    r = _get("/works", params, mailto=mailto)
    if not (r and r.status_code == 200):
        return
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
    filters = [
        "has_abstract:true",
        f"publication_year:>{year_lower_bound-1}",
        "primary_topic.field.id:fields/16|fields/25",
        f"type:{include_types}",
    ]
    results: List[Dict[str, str]] = []
    for term in query_terms:
        if len(results) >= limit:
            break
        params = {
            "search": term,
            "per_page": 200,
            "filter": ",".join(filters),
            "select": "id,title,abstract_inverted_index",
            "sort": "relevance_score:desc",
        }
        page = 1
        while len(results) < limit:
            params["page"] = page
            r = _get("/works", params, mailto=mailto)
            if not (r and r.status_code == 200):
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
    seen, uniq = set(), []
    for rec in results:
        oid = rec.get("openalex_id")
        if oid and oid not in seen:
            uniq.append(rec); seen.add(oid)
    return uniq[:limit]

# ---- interface helper ----
def fetch_by_title(title: str, mailto: Optional[str] = None) -> Optional[Dict[str, str]]:
    # 1) precise title.search
    p1 = {"filter": f"title.search:{title}", "per_page": 5, "sort": "relevance_score:desc"}
    r = _get("/works", p1, mailto=mailto)
    items = r.json().get("results", []) if (r and r.status_code == 200) else []

    # 2) broad search
    if not items:
        p2 = {"search": title, "per_page": 5, "sort": "relevance_score:desc"}
        r = _get("/works", p2, mailto=mailto)
        items = r.json().get("results", []) if (r and r.status_code == 200) else []
    if not items:
        return None

    obj = _best_title_match(title, items) or items[0]

    # ensure abstract via second fetch-by-id if needed
    inv = obj.get("abstract_inverted_index")
    if inv is None:
        r2 = _get(f"/works/{obj.get('id','')}", {"select": "abstract_inverted_index,title"}, mailto=mailto)
        if r2 and r2.status_code == 200:
            o2 = r2.json()
            inv = o2.get("abstract_inverted_index")
            obj["title"] = o2.get("title") or obj.get("title", "")

    abstract = reconstruct_abstract(inv)
    return {"title": obj.get("title") or obj.get("display_name") or "", "abstract": abstract}
