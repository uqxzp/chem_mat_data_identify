import argparse, json, random, sys, unicodedata, hashlib
from pathlib import Path
from typing import List, Dict

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "").strip()).lower()
    return " ".join(s.split())

def key_for(obj: Dict) -> str:
    t = norm(obj.get("title", ""))
    a = norm(obj.get("abstract", ""))
    return hashlib.sha1(f"{t}\n{a}".encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows: List[Dict] = []
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] {path}:{i} invalid JSON ({e}). Skipping.", file=sys.stderr)
                    continue
                rows.append(obj)

    # remove duplicates and shuffle
    seen, uniq = set(), []
    for obj in rows:
        k = key_for(obj)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(obj)
    random.shuffle(uniq)

    # write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for obj in uniq:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Merged {len(uniq)} examples into {out_path} (from {len(rows)} lines across {len(args.inputs)} files)")

if __name__ == "__main__":
    main()
