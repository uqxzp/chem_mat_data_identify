Binary classifier (title + abstract -> has dataset or not) using TinyLlama + QLoRA.



Predict with title:

```
python predict_by_title.py --model_dir outputs/classifier_448
```

Train (low VRAM settings):

```
python train_classifier.py \
  --outdir outputs/classifier_448 \
  --epochs 8 \
  --max_length 448 \
  --batch_size 1 \
  --val_pos 10 \
  --val_neg 40 \
  --test_pos 10 \
  --test_neg 40
```
