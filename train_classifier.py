import argparse
import json
import os
import random

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from utils.train_val_test_split import create_train_val_test_split
from utils.visual_helper import visualize_predictions

PROMPT_TEMPLATE = (
    "You are a classifier. Decide if the following paper releases a dataset.\n"
    "Title: {title}\n\nAbstract: {abstract}\nAnswer:"
)


def prepare_dataset(samples: list[dict]) -> Dataset:
    ds = Dataset.from_list(samples)
    return ds.map(
        lambda s: {
            "text": PROMPT_TEMPLATE.format(
                title=s.get("title"), abstract=s.get("abstract", "")
            ),
            "label": s.get("label"),
        },
        remove_columns=ds.column_names,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs/classifier_512")
    parser.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--val_pos", type=int, default=10)
    parser.add_argument("--val_neg", type=int, default=40)
    parser.add_argument("--test_pos", type=int, default=10)
    parser.add_argument("--test_neg", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)

    train_set, val_set, test_set = create_train_val_test_split(
        args.val_pos, args.val_neg, args.test_pos, args.test_neg
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
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=args.max_length
        ),
        batched=True,
        remove_columns=["text"],
    )

    model: torch.nn.Module = AutoModelForSequenceClassification.from_pretrained(
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

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_cfg)
    print("LoRA adapters active")

    """
    # print number of trainable parameters

    trainable, total = 0, 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    """

    training_args = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        report_to="none",
        lr_scheduler_type="cosine",
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return accuracy_metric.compute(
            predictions=preds, references=eval_pred.label_ids
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

    trainer.train()
    metrics = trainer.evaluate()

    predictions = trainer.predict(tokenized["test"])
    probs = torch.softmax(torch.from_numpy(predictions.predictions), dim=-1)[:, 1]
    pred_path = os.path.join(args.outdir, "test_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for row, prob in zip(test_set, probs.tolist()):
            f.write(
                json.dumps(
                    {
                        "label": row.get("label"),
                        "prediction": prob,
                        "title": row.get("title"),
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
