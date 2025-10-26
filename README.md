Binary classifier (title + abstract -> has dataset or not) using TinyLlama + QLoRA.

No dependencies on ChemMatData, can be easily merged later.

Train (low VRAM):

```
python src/train_classifier.py \
  --train_path data/dataset_train.jsonl \
  --test_path  data/dataset_test.jsonl \
  --outdir outputs/r5_r8_qkvo_len224 \
  --epochs 5 \
  --max_length 224 \
  --use_4bit \
  --grad_checkpointing \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_targets q_proj k_proj v_proj o_proj \
  --optim paged_adamw_8bit
```

Evaluate on test set (10 positives and negatives each)

```
python src/eval_on_test.py \
  --model_dir outputs/r5_r8_qkvo_len224 \
  --test_path data/dataset_test.jsonl \
  --max_length 224 \
  --batch_size 1 \
  --use_4bit \
  --pred_out outputs/preds.jsonl
```
