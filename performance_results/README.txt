Results

Test set: data/test_set.jsonl
50 positives from Pascal's list
50 random negatives 

Threshold 0.05

Qwen (~1.5B) 
26/50 positives > 0.05
1/50 negatives > 0.05
Ratio TP/FN 26:1

tinyllama + QLoRA (~500M)
47/50 positives > 0.05
3/50 negatives > 0.05
Ratio TP/FN 15.67:1