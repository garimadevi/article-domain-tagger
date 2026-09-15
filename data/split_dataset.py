"""Frozen 80/10/10 stratified split of the v2 mapped corpus.

Reads generated/news_domain_tagging_v2.csv, splits on `label` with
SPLIT_SEED=42, writes split_train.csv / split_val.csv / split_test.csv under
generated/. Idempotent: rerunning with the same seed reproduces identical files.

Usage:
  python -m data.split_dataset
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def main():
    df = pd.read_csv(config.V2_CSV)
    print(f"[split] rows: {len(df)}")

    train, temp = train_test_split(
        df, test_size=(1 - config.TRAIN_RATIO), random_state=config.SPLIT_SEED, stratify=df["label"]
    )
    val, test = train_test_split(
        temp,
        test_size=config.VALID_RATIO / (1 - config.TRAIN_RATIO),
        random_state=config.SPLIT_SEED,
        stratify=temp["label"],
    )

    for name, out, part in (("train", config.TRAIN_CSV, train), ("val", config.VAL_CSV, val), ("test", config.TEST_CSV, test)):
        part = part.sort_values("label").reset_index(drop=True)
        part.to_csv(out, index=False)
        print(f"[split] {name:5s} -> {out}  ({len(part)} rows)")
        counts = part["label"].value_counts().sort_index()
        print("         label counts:", dict(counts))

    assert len(train) + len(val) + len(test) == len(df)
    print(f"[split] total check OK: {len(train)}+{len(val)}+{len(test)} = {len(df)}")


if __name__ == "__main__":
    main()