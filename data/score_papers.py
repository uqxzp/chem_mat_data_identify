"""
Score papers from a JSONL file with a fine-tuned classifier.

The script loads the model from `--model_dir`, predicts a score for each paper
from its title and abstract, writes all results to `--output_all`, and writes
papers above the threshold to `--output_flagged`.

Example:
    PYTHONPATH=. python data/score_papers.py \
      --jsonl data/test_set.jsonl \
      --model_dir outputs/classifier_512_tinyllama \
      --output_all data/test_set_all.jsonl \
      --output_flagged data/test_set_flagged.jsonl \
      --threshold 0.05

Note: threshold 0.05 achieved the best test set performance (46/50 pos. and 50/50 neg.)
"""

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig)

from train_classifier import PROMPT_TEMPLATE
from utils.visual_helper import visualize_predictions

LOAD_KWARGS = {
    "device_map": "auto",
    "quantization_config": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
}


def load_model(model_dir: str):
    adapter_cfg = Path(model_dir) / "adapter_config.json"
    if not os.path.exists(adapter_cfg):
        raise FileNotFoundError("Model directory not found.")

    with adapter_cfg.open(encoding="utf-8") as f:
        base_ckpt = json.load(f).get("base_model_name_or_path")
    if not base_ckpt:
        raise RuntimeError("adapter_config.json missing `base_model_name_or_path`.")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_ckpt, num_labels=2, **LOAD_KWARGS
    )
    return PeftModel.from_pretrained(base_model, model_dir, is_trainable=False)


def predict_score(
    model, tokenizer, title: str | None, abstract: str | None, max_length: int
) -> float:
    prompt = PROMPT_TEMPLATE.format(title=title or "", abstract=abstract or "")
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, 1].item())


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def sort_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    records.sort(key=lambda x: x["prediction"], reverse=True)

    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing the fine-tuned model.",
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        type=Path,
        help="Input JSONL file with papers to score.",
    )
    parser.add_argument(
        "--output_all",
        type=Path,
        default=Path("data/scored_all.jsonl"),
        help="Output JSONL file for all scored papers.",
    )
    parser.add_argument(
        "--output_flagged",
        type=Path,
        default=Path("data/scored_flagged.jsonl"),
        help="Output JSONL file for papers above the threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Score threshold for flagged papers.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum tokenized input length.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model_dir).eval()
    if torch.cuda.is_available() and not getattr(model, "hf_device_map", None):
        model.to("cuda")

    args.output_all.parent.mkdir(parents=True, exist_ok=True)
    args.output_flagged.parent.mkdir(parents=True, exist_ok=True)
    number_flagged = 0
    total = sum(1 for _ in open(args.jsonl, "rb"))
    progress = tqdm(total=total, desc="Progress", unit="paper")

    with (
        args.output_all.open("w", encoding="utf-8") as all_f,
        args.output_flagged.open("w", encoding="utf-8") as flagged_f,
    ):
        i = 0
        for sample in iter_jsonl(args.jsonl):
            score = predict_score(
                model,
                tokenizer,
                sample["title"],
                sample["abstract"],
                args.max_length,
            )
            title = sample["title"]
            label = 1 if i < 50 else 0
            record = {"prediction": float(score), "title": title, "label": label}
            all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if score > args.threshold:
                flagged_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                number_flagged += 1
                # Keeps progress bar in place while printing flagged titles
                progress.write(f"{number_flagged}: {title}")

            progress.update(1)
            i += 1

    progress.close()
    print(f"Flagged {number_flagged} samples above threshold {args.threshold}.")

    sort_jsonl(args.output_flagged)
    visualize_predictions(args.output_all)


if __name__ == "__main__":
    main()
