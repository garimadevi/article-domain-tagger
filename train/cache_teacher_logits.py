"""Cache teacher logits (soft targets) for all train-split rows.

Runs the revised 10-label teacher over the frozen train split once and saves
logits to NPZ (optionally fp16). The student uses these cached logits for
logit-KL distillation so the teacher never needs to run during student training.

Designed for Colab T4:
  python -m train.cache_teacher_logits \
    --teacher "<Drive>/distilbert_teacher_revised_10_labels" \
    --data-dir generated --out generated/teacher_logits

Smoke test (local CPU):
  python -m train.cache_teacher_logits --teacher distilbert-base-uncased \
      --sample 200 --batch-size 8 --out generated/teacher_logits_smoke
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from train.common import add_common_args, device_from_args, load_split, make_loaders


@torch.no_grad()
def cache(model, loader, device, fp16=False):
    model.eval()
    logits_list = []
    idx_list = []
    start = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits.float()
        if fp16:
            logits = logits.half()
        logits_list.append(logits.cpu().numpy())
        idx_list.append(np.arange(start, start + len(batch["label"])))
        start += len(batch["label"])
    logits = np.concatenate(logits_list, axis=0)
    idx = np.concatenate(idx_list, axis=0)
    return logits, idx


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--teacher", default=str(config.TEACHER_REVISED))
    ap.add_argument("--out", default=str(config.TEACHER_LOGITS))
    ap.add_argument("--fp16", action="store_true", help="store logits as float16 (halves file size)")
    args = ap.parse_args()

    device = device_from_args(args)
    print(f"[cache] device={device} teacher={args.teacher}")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher)
    model.to(device)

    train_df = load_split(args.data_dir, "train", args.sample)
    loader = make_loaders(train_df, tokenizer, args.max_len, args.batch_size, args.workers,
                          shuffle=False, device=str(device))
    print(f"[cache] caching logits for {len(train_df)} train rows ...")

    logits, idx = cache(model, loader, device, fp16=args.fp16)
    print(f"[cache] logits shape {logits.shape} dtype {logits.dtype}")

    os.makedirs(args.out, exist_ok=True)
    np.savez(f"{args.out}/train_logits.npz", logits=logits, indices=idx)
    print(f"[cache] saved -> {args.out}/train_logits.npz")

    # sanity: logits[i] corresponds to split_train row i (row order preserved)
    assert len(idx) == len(train_df)
    assert idx[0] == 0 and int(idx[-1]) == len(train_df) - 1
    print("[cache] index alignment OK (row i of split_train.csv <-> logits[i])")


if __name__ == "__main__":
    main()