import torch
import os, json, argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from peft import PeftModel
from tools.fetch_openalex import fetch_by_title

import warnings
from transformers.utils import logging as hf_logging
import logging as pylog

# run with python -m src.predict_by_title --model_dir outputs/r5_r8_qkvo_len224

def load_model(model_dir: str, use_4bit: bool=True):
    kw={}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"]=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            kw["device_map"]="auto"
        except Exception: pass

    cfg_path=os.path.join(model_dir,"adapter_config.json")
    if os.path.exists(cfg_path):
        base=json.load(open(cfg_path))["base_model_name_or_path"]
        cfg=AutoConfig.from_pretrained(base); cfg.num_labels=2; cfg.problem_type="single_label_classification"
        base_m=AutoModelForSequenceClassification.from_pretrained(base,config=cfg,**kw)
        return PeftModel.from_pretrained(base_m, model_dir)
    return AutoModelForSequenceClassification.from_pretrained(model_dir,**kw)

def predict_one(model, tok, title, abstract, max_length=224):
    x=tok(text=title or "", text_pair=abstract or "", truncation="only_second", max_length=max_length, return_tensors="pt")
    x={k:v.to(model.device) for k,v in x.items()}
    with torch.no_grad():
        return float(torch.softmax(model(**x).logits, dim=-1)[0,1].item())

def main():
    # low vram
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF","expandable_segments:True,max_split_size_mb:64")

    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_length", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--use_4bit", action="store_true", default=True) # change default after development?
    args=ap.parse_args()

    # hide transformers warnings
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    pylog.getLogger("transformers").setLevel(pylog.ERROR)
    pylog.getLogger("peft").setLevel(pylog.ERROR)
    pylog.getLogger("bitsandbytes").setLevel(pylog.ERROR)

    # tokenizer... 
    tok=AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=load_model(args.model_dir, use_4bit=args.use_4bit).eval()
    if torch.cuda.is_available(): model.to("cuda")

    print("Enter title (empty to quit):")
    while True:
        try: title=input("> ").strip()
        except (EOFError,KeyboardInterrupt): break
        if not title: break

        rec = fetch_by_title(title)
        if not rec:
            print("No match on OpenAlex. Paste abstract (or Enter to skip):")
            abs_manual = input().strip()
            if not abs_manual: continue
            p = predict_one(model, tok, title, abs_manual, args.max_length)
        else:
            if not rec["abstract"]:
                print("[note] No abstract via API; predicting from title only.")
            p = predict_one(model, tok, rec["title"], rec["abstract"], args.max_length)

        print(f"Prediction: {'Yes' if p>=args.threshold else 'No'}. Score: {p:.2f}")

if __name__=="__main__": main()
