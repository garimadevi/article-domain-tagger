"""Final evaluation of any trained checkpoint on the held-out test split.

Reports accuracy, macro-F1, per-class F1 + precision/recall, confusion matrix
(row-major as CSV + optional PNG), and the FLOP budget re-check.

Usage:
  python -m eval.evaluate --model <dir> [--out generated/eval_report]
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

import config
from train.common import load_split, make_loaders


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, true = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        preds.append(out.logits.argmax(-1).cpu().numpy())
        true.append(batch["label"].numpy())
    return np.concatenate(preds), np.concatenate(true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default=str(config.GENERATED))
    ap.add_argument("--out", default=str(config.EVAL_REPORT))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=config.MAX_LEN)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--flops", action="store_true", help="re-measure FLOPs on exported model")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)

    test_df = load_split(args.data_dir, "test")
    loader = make_loaders(test_df, tokenizer, args.max_len, args.batch_size, args.workers,
                          shuffle=False, device=str(device))
    preds, true = predict(model, loader, device)
    print(f"[eval] test rows={len(test_df)} acc={accuracy_score(true, preds):.4f} "
          f"f1_macro={f1_score(true, preds, average='macro'):.4f}")

    labels = config.LABELS
    p, r, f, _ = precision_recall_fscore_support(true, preds, labels=list(range(len(labels))), zero_division=0)
    df_report = pd.DataFrame({
        "label": labels,
        "count": pd.Series(true).value_counts().reindex(range(len(labels))).fillna(0).astype(int).values,
        "precision": p.round(4),
        "recall": r.round(4),
        "f1": f.round(4),
    })
    print(df_report.to_string(index=False))

    os.makedirs(args.out, exist_ok=True)
    df_report.to_csv(f"{args.out}/per_class.csv", index=False)
    cm = confusion_matrix(true, preds, labels=list(range(len(labels))))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(f"{args.out}/confusion_matrix.csv")
    print(f"[eval] wrote -> {args.out}/per_class.csv, confusion_matrix.csv")

    try:  # matplotlib optional
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        norm_cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(norm_cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                        color="white" if norm_cm[i, j] > 0.5 else "black", fontsize=7)
        ax.set_title(f"Confusion matrix (normalized) — macro-F1 {f1_score(true, preds, average='macro'):.3f}")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(f"{args.out}/confusion_matrix.png", dpi=150)
        print(f"[eval] wrote -> {args.out}/confusion_matrix.png")
    except Exception as e:
        print(f"[eval] skipped confusion plot: {e}")

    if args.flops:
        from analysis.measure_flops import measure_flops
        model.eval()
        with torch.no_grad():
            flops_rows = [x for x in measure_flops(model, [args.max_len], device=str(device))][0]
        print(f"[eval] FLOP re-check @{flops_rows[0]}: {flops_rows[1]/1e9:.3f} GFLOP "
              f"(gate 1.0) -> {'PASS' if flops_rows[1]/1e9 <= config.GFLOP_BUDGET else 'FAIL'}")


if __name__ == "__main__":
    main()