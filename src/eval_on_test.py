from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, Any, List
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

# python src/eval_on_test.py --model_dir outputs/[model] --test_path data/dataset_test.jsonl 
# --max_length [224/256] --batch_size 1 --use_4bit --pred_out outputs/[output file]

# ---- helpers ----

def format_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    ex["title"] = ex.get("title") or ""
    ex["abstract"] = ex.get("abstract") or ""
    ex["label"] = int(ex.get("label", 0))
    return ex

def tokenize_pair(tok, titles, abstracts, max_length: int):
    return tok(
        text=titles,
        text_pair=abstracts,
        truncation="only_second",  # keep full title
        max_length=max_length,
    )

def load_model_with_adapters(model_dir: str, use_4bit: bool):
    load_kwargs = {}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
            print("Using 4-bit inference.")
        except Exception as e:
            print("Warning: ", e)

    adapter_cfg_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            aconf = json.load(f)
        base_name = aconf.get("base_model_name_or_path", None)
        if not base_name:
            print("[error] adapter_config.json missing base_model_name_or_path")
            sys.exit(1)
        print("Base model:", base_name)

        base_cfg = AutoConfig.from_pretrained(base_name)
        base_cfg.num_labels = 2
        base_cfg.problem_type = "single_label_classification"

        base = AutoModelForSequenceClassification.from_pretrained(
            base_name, config=base_cfg, **load_kwargs
        )

        try:
            from peft import PeftModel
            print("Attaching PEFT adapters from:", model_dir)
            model = PeftModel.from_pretrained(base, model_dir)
            return model
        except Exception as e:
            print("[error] Could not attach adapters:", e)
            sys.exit(1)

    print("Loading plain HF checkpoint from:", model_dir)
    return AutoModelForSequenceClassification.from_pretrained(model_dir, **load_kwargs)

# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--max_length", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--use_4bit", action="store_true", help="Low-VRAM 4-bit inference if bitsandbytes is available")
    ap.add_argument("--pred_out", default="predictions_test.jsonl")
    args = ap.parse_args()

    # dataset
    ds = load_dataset("json", data_files=args.test_path, split="train")
    ds = ds.map(format_example)

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # tokenize; truncate abstract first
    def _tok(batch):
        return tokenize_pair(tok, batch["title"], batch["abstract"], args.max_length)
    ds_tok = ds.map(_tok, batched=True)

    # model
    model = load_model_with_adapters(args.model_dir, args.use_4bit)
    model.eval()
    if not hasattr(model, "device") or str(model.device) == "cpu":
        if torch.cuda.is_available():
            model.to("cuda")

    # metrics
    metric_acc  = evaluate.load("accuracy")
    metric_f1   = evaluate.load("f1")
    metric_prec = evaluate.load("precision")
    metric_rec  = evaluate.load("recall")
    metric_auc  = evaluate.load("roc_auc", "binary")

    # batched prediction
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
    cols = ["input_ids", "attention_mask"]

    def iter_batches(dataset, batch_size: int):
        batch = []
        for ex in dataset:
            batch.append(ex)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    all_probs: List[List[float]] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in iter_batches(ds_tok, max(1, args.batch_size)):
            # collate
            inputs = {k: [ex[k] for ex in batch] for k in cols}
            inputs = collator(inputs)
            for k in inputs:
                inputs[k] = inputs[k].to(model.device)

            logits = model(**inputs).logits  # [B, 2]
            # stable softmax
            m = logits.max(dim=1, keepdim=True).values
            probs = torch.exp(logits - m) / torch.exp(logits - m).sum(dim=1, keepdim=True)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend([int(ex["label"]) for ex in batch])

    # metrics dict
    probs_np = np.array(all_probs)
    y_true   = np.array(all_labels)
    y_pred   = probs_np.argmax(axis=1)

    results = {}
    results.update(metric_acc.compute(predictions=y_pred, references=y_true))
    results.update(metric_f1.compute(predictions=y_pred, references=y_true, average="binary"))
    results.update(metric_prec.compute(predictions=y_pred, references=y_true, average="binary"))
    results.update(metric_rec.compute(predictions=y_pred, references=y_true, average="binary"))
    try:
        results["roc_auc"] = metric_auc.compute(
            prediction_scores=probs_np[:, 1].tolist(), references=y_true.tolist()
        )["roc_auc"]
    except Exception:
        results["roc_auc"] = float("nan")

    print(json.dumps(results, indent=2))

    with open(args.pred_out, "w", encoding="utf-8") as f:
        for ex, p in zip(ds, probs_np):
            out = {
                "title": ex["title"],
                "abstract": ex["abstract"],
                "label": int(ex["label"]),
                "p_no_dataset": float(p[0]),
                "p_has_dataset": float(p[1]),
                "pred": int(int(p[1] >= p[0])),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f">> Wrote per-row predictions to {args.pred_out}")

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
    main()
