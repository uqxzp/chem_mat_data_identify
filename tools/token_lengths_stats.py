import json, argparse, numpy as np
from transformers import AutoTokenizer

# python tools/token_length_stats.py --jsonl data/dataset_test/train.jsonl
# mean title + abstracts ca. 280, 224 might perform better than 256 due to slice shift resulting in more boilerplate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="e.g. data/dataset_test.jsonl")
    ap.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    pair_lengths, abs_lengths, title_lengths = [], [], []

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            title = ex.get("title") or ""
            abstract = ex.get("abstract") or ""

            pair_ids  = tok(text=title, text_pair=abstract, truncation=False)["input_ids"]
            abs_ids   = tok(abstract, truncation=False, add_special_tokens=True)["input_ids"]
            title_ids = tok(title, truncation=False, add_special_tokens=True)["input_ids"]

            pair_lengths.append(len(pair_ids))
            abs_lengths.append(len(abs_ids))
            title_lengths.append(len(title_ids))

    def stats(arr):
        a = np.array(arr)
        q = lambda p: int(np.percentile(a, p)) if a.size else 0
        return {
            "count": int(a.size),
            "min": int(a.min()) if a.size else 0,
            "p50": q(50),
            "p90": q(90),
            "max": int(a.max()) if a.size else 0,
            "mean": float(a.mean()) if a.size else 0.0,
        }

    out = {
        "title_plus_abstract_tokens": stats(pair_lengths),
        "abstract_only_tokens": stats(abs_lengths),
        "title_only_tokens": stats(title_lengths),
        "required_max_length_for_full_title_plus_abstract":
            int(max(pair_lengths) if pair_lengths else 0),
    }

    import json as _json
    print(_json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
