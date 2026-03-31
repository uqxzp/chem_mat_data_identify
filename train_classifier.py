"""
Start a new training run:

PYTHONPATH=. python train_classifier.py \
  --outdir outputs/classifier_512_vXXX

Resume training from saved LoRA adapters and a trainer checkpoint:
- Set epochs to total number of epochs, not just additional ones.
- Adjust learning rate if necessary.

PYTHONPATH=. python train_classifier.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --outdir outputs/classifier_512_vXXX \
  --resume_from outputs/classifier_512_vYYY \
  --resume_checkpoint outputs/classifier_512_vYYY/checkpoint-ZZZZ

Notes:
- Hyperparameters are defined as constants in this script.
- Replace placeholders in the example commands.
"""

import argparse
import json
import os

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from utils.data_loader import load_train_val_test
from utils.visual_helper import visualize_predictions

# Training hyperparameters
MAX_TOKENS = 512
TRAINING_EPOCHS = 8
TRAINING_BATCH_SIZE = 1
TRAINING_LR = 3e-5
TRAINING_VAL_POS = 25
TRAINING_VAL_NEG = 25
TRAINING_SEED = 1234
TRAINING_GRAD_ACC = 4
TRAINING_LOGGING_STEPS = 50

# QLoRA
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

PROMPT_TEMPLATE = """You are a classifier. Decide if the following paper releases a dataset.
Title: {title}
Abstract: {abstract}
Answer:"""


def prepare_dataset(samples: list[dict]) -> Dataset:
    ds = Dataset.from_list(samples)

    def transform(s: dict) -> dict:
        return {
            "text": PROMPT_TEMPLATE.format(title=s["title"], abstract=s["abstract"]),
            "label": s["label"],
        }

    return ds.map(transform, remove_columns=ds.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", required=True, help="Output directory of the fine-tuned model."
    )
    parser.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--resume_from")
    parser.add_argument("--resume_checkpoint")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(TRAINING_SEED)
    train_set, val_set, test_set = load_train_val_test(
        TRAINING_VAL_POS, TRAINING_VAL_NEG
    )

    dataset = DatasetDict(
        train=prepare_dataset(train_set),
        validation=prepare_dataset(val_set),
        test=prepare_dataset(test_set),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:  # for batch size > 1
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_TOKENS),
        batched=True,
        remove_columns=["text"],
    )

    base_model: torch.nn.Module = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map="auto",
    )

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        modules_to_save=["score"],
    )
    if args.resume_from:
        model = PeftModel.from_pretrained(
            base_model, args.resume_from, is_trainable=True
        )
        print(f"Loaded adapters from {args.resume_from}")
    else:
        model = get_peft_model(base_model, lora_cfg)
        print("LoRA adapters active")

    training_args = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=TRAINING_BATCH_SIZE,
        per_device_eval_batch_size=TRAINING_BATCH_SIZE,
        learning_rate=TRAINING_LR,
        num_train_epochs=TRAINING_EPOCHS,
        gradient_accumulation_steps=TRAINING_GRAD_ACC,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=TRAINING_LOGGING_STEPS,
        lr_scheduler_type="cosine",
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        class_preds = np.argmax(eval_pred.predictions, axis=-1)
        return accuracy_metric.compute(
            predictions=class_preds, references=eval_pred.label_ids
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    metrics = trainer.evaluate()

    predictions = trainer.predict(tokenized["test"])
    preds = torch.softmax(torch.from_numpy(predictions.predictions), dim=-1)[:, 1]
    pred_path = os.path.join(args.outdir, "test_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for row, pred in zip(test_set, preds.tolist()):
            f.write(
                json.dumps(
                    {
                        "label": row["label"],
                        "prediction": pred,
                        "title": row["title"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    trainer.save_model(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    visualize_predictions(pred_path)


if __name__ == "__main__":
    main()
