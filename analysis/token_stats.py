"""Token-length statistics for the taxonomy-v2 corpus using the target tokenizer.

Loads the mapped v2 CSV, tokenizes all texts with distilbert-base-uncased
(offline, from HF cache) and reports the length distribution plus truncation
impact at candidate max_lengths. Read-only over generated/v2; writes nothing
except a stats report under generated/ (optional).

Usage:
  python -m analysis.token_stats [--out generated/token_stats.csv]
"""

import argparse

import numpy as np
import pandas as pd

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=config.GENERATED / "token_stats.csv")
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    df = pd.read_csv(config.V2_CSV)
    print(f"[token_stats] rows: {len(df)}")

    n = len(df)
    tokens = tokenizer(
        df["text"].fillna("").astype(str).tolist(),
        truncation=False,
        add_special_tokens=True,
    )
    lens = np.array([len(x) for x in tokens["input_ids"]], dtype=np.int32)

    quantiles = {}
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        quantiles[f"p{int(q*100)}"] = int(np.quantile(lens, q))
    stats = pd.DataFrame([{"n": n, "max": int(lens.max()), "mean": float(lens.mean()), **quantiles}])

    trunc = []
    for ml in (32, 48, 64, 96, 128):
        trunc.append({"max_len": ml, "truncated_frac": float((lens > ml).mean())})
    trunc_df = pd.DataFrame(trunc)

    print("\n[n] =", n)
    print(stats.T.to_string())
    print("\ntruncation impact per max_len:")
    print(trunc_df.to_string(index=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.concat(
            [stats.assign(stat="summary"), trunc_df.assign(stat="truncation")],
            ignore_index=True,
        )
        df_out.to_csv(args.out, index=False)
        print(f"\n[tok] report -> {args.out}")


if __name__ == "__main__":
    main()