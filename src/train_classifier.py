from __future__ import annotations
import os, math, argparse, json, numpy as np
from typing import Dict, Any
from dataclasses import fields, is_dataclass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

"""
low vram command for training

python src/train_classifier.py \
  --train_path data/dataset_train.jsonl \
  --test_path  data/dataset_test.jsonl \
  --outdir outputs/r5_r8_qkvo_len224_test \
  --epochs 5 \
  --max_length 224 \
  --use_4bit \
  --grad_checkpointing \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_targets q_proj k_proj v_proj o_proj \
  --optim paged_adamw_8bit
"""

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments
)

# --- helpers ---
def build_training_args_relaxed(**kwargs) -> TrainingArguments:
    # drop kwargs unsupported by local transformers version
    allowed = None
    try:
        if is_dataclass(TrainingArguments):
            allowed = {f.name for f in fields(TrainingArguments)}
    except Exception:
        pass
    if not allowed:
        import inspect
        allowed = {p.name for p in inspect.signature(TrainingArguments.__init__).parameters.values()}
    pruned = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**pruned)

def format_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    title = ex.get("title") or ""
    abstract = ex.get("abstract") or ""
    ex["text"] = f"Title: {title}\n\nAbstract: {abstract}"
    ex["label"] = int(ex.get("label", 0))
    return ex

# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--test_path",  required=True)
    ap.add_argument("--outdir",     default="outputs/tinyllama_cls")
    ap.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # core knobs
    ap.add_argument("--max_length", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs",     type=int, default=1)
    ap.add_argument("--lr",         type=float, default=2e-4)

    # vram
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--grad_checkpointing", action="store_true")

    # lora
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=2)
    ap.add_argument("--lora_alpha", type=int, default=4)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", nargs="+", default=["v_proj"])

    # optimizer
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # data
    train = load_dataset("json", data_files=args.train_path, split="train")
    test  = load_dataset("json", data_files=args.test_path,  split="train")
    ds = DatasetDict(train=train.map(format_example), validation=test.map(format_example))

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)
    keep_cols = ["label"]
    ds = ds.map(tokenize, batched=True,
                remove_columns=[c for c in ds["train"].column_names if c not in keep_cols])

    # model
    cfg = AutoConfig.from_pretrained(args.model_name, num_labels=2, problem_type="single_label_classification")
    load_kwargs = {}
    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import torch
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
            print("4-bit quantization on")
        except Exception as e:
            print("bitsandbytes unavailable:", e)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=cfg, **load_kwargs)
    try: model.config.use_cache = False
    except Exception: pass
    if args.grad_checkpointing:
        try: model.gradient_checkpointing_enable(); print("gradient checkpointing on")
        except Exception: pass

    if args.no_lora:
        print("head-only training (no LoRA)")
    else:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            peft_cfg = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                target_modules=args.lora_targets, bias="none",
            )
            model = get_peft_model(model, peft_cfg)
            try: model.print_trainable_parameters()
            except Exception: pass
        except Exception as e:
            print("[warn] PEFT unavailable; proceeding without LoRA:", e)

    # keep vram low
    steps_per_epoch = max(1, math.ceil(len(ds["train"]) / max(1, args.batch_size) / max(1, args.grad_accum)))
    targs = build_training_args_relaxed(
        output_dir=args.outdir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_steps=max(200, steps_per_epoch),
        logging_steps=50,
        report_to="none",
        optim=args.optim,
        dataloader_num_workers=0,
    )

    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=None,
        tokenizer=tok, data_collator=collator,
        compute_metrics=None,
    )

    trainer.train()
    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)
    with open(os.path.join(args.outdir, "label_map.json"), "w") as f:
        json.dump({"0": "no-dataset", "1": "has-dataset"}, f)

    print("saved:", args.outdir)

if __name__ == "__main__":
    main()
