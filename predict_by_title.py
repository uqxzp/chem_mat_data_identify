import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig)

from train_classifier import PROMPT_TEMPLATE
from utils.fetch_openalex import fetch_by_title


def load_model(model_dir: str):
    adapter_cfg = Path(model_dir) / "adapter_config.json"
    load_kwargs = {
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
    }

    if adapter_cfg.exists():
        with open(adapter_cfg, encoding="utf-8") as f:
            base_ckpt = json.load(f).get("base_model_name_or_path")
        if not base_ckpt:
            raise RuntimeError("adapter_config.json missing base_model_name_or_path")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_ckpt, num_labels=2, **load_kwargs
        )
        return PeftModel.from_pretrained(base_model, model_dir, is_trainable=False)

    return AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=2, **load_kwargs
    )


def predict_one(model, tokenizer, title: str | None, abstract: str | None, max_length: int):
    prompt = PROMPT_TEMPLATE.format(title=title, abstract=abstract)
    encoded = tokenizer(
        prompt, truncation=True, max_length=max_length, return_tensors="pt"
    )
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits
    return torch.softmax(logits, dim=-1)[0, 1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--max_length", type=int, default=448)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model_dir).eval()
    if torch.cuda.is_available() and not getattr(model, "hf_device_map", None):
        model.to("cuda")
    
    print("Enter a publication title (empty line to quit):")
    while True:
        try:
            query = input("> ").strip()
            if not query:
                break
        except (EOFError, KeyboardInterrupt):
            break

        record = fetch_by_title(query)
        if not record:
            print("Publication not found.")
            continue
        score = predict_one(
            model,
            tokenizer,
            title=record.get("title") or query,
            abstract=record.get("abstract"),
            max_length=args.max_length,
        )
        print(record.get("title"))
        print(f"Score: {score:.3f}")


if __name__ == "__main__":
    main()
