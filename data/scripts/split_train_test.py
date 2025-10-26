import argparse, json, random, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out-train", required=True)
ap.add_argument("--out-test", required=True)
ap.add_argument("--n-pos-test", type=int, default=10)
ap.add_argument("--n-neg-test", type=int, default=10)
args = ap.parse_args()

pos, neg = [], []

# read + bucket
with open(args.inp, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[WARN] skip line {i}: {e}", file=sys.stderr)
            continue
        if "label" not in obj:
            print(f"[WARN] skip line {i}: missing 'label'", file=sys.stderr)
            continue
        try:
            lbl = int(obj["label"])
        except Exception:
            print(f"[WARN] skip line {i}: non-integer label", file=sys.stderr)
            continue
        if lbl == 1:
            pos.append(obj)
        elif lbl == 0:
            neg.append(obj)
        else:
            print(f"[WARN] skip line {i}: label must be 0/1", file=sys.stderr)

# ensure enough
if len(pos) < args.n_pos_test or len(neg) < args.n_neg_test:
    print(
        f"[ERROR] Not enough samples. Have pos={len(pos)}, neg={len(neg)}; "
        f"need at least {args.n_pos_test} pos and {args.n_neg_test} neg.",
        file=sys.stderr,
    )
    sys.exit(1)

random.shuffle(pos)
random.shuffle(neg)

test = pos[:args.n_pos_test] + neg[:args.n_neg_test]
train = pos[args.n_pos_test:] + neg[args.n_neg_test:]

random.shuffle(test)
random.shuffle(train)

# write
Path(Path(args.out_train).parent).mkdir(parents=True, exist_ok=True)
Path(Path(args.out_test).parent).mkdir(parents=True, exist_ok=True)

with open(args.out_test, "w", encoding="utf-8") as ft:
    for obj in test:
        ft.write(json.dumps(obj, ensure_ascii=False) + "\n")

with open(args.out_train, "w", encoding="utf-8") as fr:
    for obj in train:
        fr.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Wrote {len(train)} train examples -> {args.out_train}")
print(f"Wrote {len(test)} test examples  -> {args.out_test}")
