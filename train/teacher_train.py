"""Retrain the synced DistilBERT checkpoint on the new 10-label taxonomy.

Loads the existing fine-tuned 8-label checkpoint (from Drive) and replaces
only the classifier head -> 10 labels, then fine-tunes with class-weighted
cross-entropy on the frozen train/val splits. The old checkpoint is never
modified; the new teacher is saved separately.

Designed for Colab T4:
  from google.colab import drive; drive.mount('/content/drive')
  python -m train.teacher_train --teacher-from "<Drive 8-label ckpt>" \
      --out "/content/drive/MyDrive/ARTICLE DOMAIN TAGGING /distilbert_teacher_revised_10_labels"

Smoke test (local CPU, tiny sample):
  python -m train.teacher_train --teacher-from distilbert-base-uncased \
      --out generated/teacher_smoke --sample 500 --epochs 1 --batch-size 8
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

import config
from train.common import (
    add_common_args, device_from_args, class_weights, load_split, make_loaders,
)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(-1).cpu().numpy()
    y = labels.cpu().numpy()
    return {
        "acc": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
        "f1_weighted": f1_score(y, preds, average="weighted"),
    }


def train_epoch(model, loader, opt, device, weights):
    model.train()
    total, running = 0, 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        opt.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attn)
        loss = torch.nn.functional.cross_entropy(out.logits, label, weight=weights.to(device))
        loss.backward()
        opt.step()
        running += loss.item() * label.size(0)
        total += label.size(0)
    return running / max(total, 1)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        all_logits.append(out.logits)
        all_labels.append(batch["label"])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return compute_metrics(logits, labels)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--teacher-from", default=str(config.TEACHER_OLD))
    ap.add_argument("--out", default=str(config.TEACHER_REVISED))
    ap.add_argument("--weights-scheme", choices=["inverse"], default="inverse")
    ap.add_argument("--save-every-epoch", action="store_true")
    args = ap.parse_args()

    device = device_from_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[teacher] device={device} teacher_from={args.teacher_from} out={args.out}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_from,
        num_labels=len(config.LABELS),
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_df = load_split(args.data_dir, "train", args.sample)
    val_df = load_split(args.data_dir, "val", args.sample)
    print(f"[teacher] train={len(train_df)} val={len(val_df)}")

    weights = class_weights(train_df["label"])
    print(f"[teacher] class weights: {weights.tolist()}")

    train_loader = make_loaders(train_df, tokenizer, args.max_len, args.batch_size, args.workers,
                                shuffle=True, device=str(device))
    val_loader = make_loaders(val_df, tokenizer, args.max_len, args.batch_size, args.workers,
                              shuffle=False, device=str(device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device, weights)
        m = eval_model(model, val_loader, device)
        print(f"[teacher] epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
              f"acc={m['acc']:.4f} f1_macro={m['f1_macro']:.4f} f1_w={m['f1_weighted']:.4f}")
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            os.makedirs(args.out, exist_ok=True)
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"[teacher] new best f1_macro={best_f1:.4f} -> saved {args.out}")
        elif args.save_every_epoch:
            ep_dir = f"{args.out}_ep{epoch}"
            os.makedirs(ep_dir, exist_ok=True)
            model.save_pretrained(ep_dir)
            tokenizer.save_pretrained(ep_dir)

    print(f"[teacher] done. best val f1_macro={best_f1:.4f}")


if __name__ == "__main__":
    main()