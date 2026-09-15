"""Load the IAB News Classification dataset, filter to political categories,
extract Environment articles from Science, map to 11 labels, and create splits.

Usage:
  python -m data.load_iab_dataset [--sample N]

Writes to generated/:
  - iab_news_v2.csv         (all mapped rows: text + label + iab_category)
  - class_counts_iab.csv    (per-label counts + shares)
  - split_train.csv         (80%)
  - split_val.csv           (10%)
  - split_test.csv          (10%)
"""

import argparse
import sys

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import config
from data import taxonomy


# IAB categories we keep (mapped to our 11 labels).
KEEP_IAB = {
    "Politics",
    "War and Conflicts",
    "Crime",
    "Law",
    "Business and Finance",
    "Technology & Computing",
    "Medical Health",
    "Education",
    "Disasters",
    "Sports",
    "Science",  # will be split into Science & Technology + Environment & Climate
}


def load_iab_frame(sample: int = 0) -> pd.DataFrame:
    print("[load] downloading IAB dataset from HuggingFace ...")
    ds = load_dataset(config.HF_DATASET, split="train")
    df = pd.DataFrame(ds)
    print(f"[load] raw rows: {len(df)}")
    if sample:
        df = df.sample(min(sample, len(df)), random_state=config.SPLIT_SEED).reset_index(drop=True)
        print(f"[load] sampled to {len(df)} rows for smoke test")
    return df


def filter_and_map(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to KEEP_IAB categories and map to our 11 labels."""
    # Filter to known IAB categories.
    df = df[df["iab_category"].isin(KEEP_IAB)].copy()
    print(f"[filter] rows after keeping IAB categories: {len(df)}")

    # Rename column for clarity.
    df = df.rename(columns={"maintext": "text", "iab_category": "iab_category_raw"})

    # Clean text.
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 50].copy()  # drop very short texts
    print(f"[filter] rows after text cleaning: {len(df)}")

    # Map Science articles: split into Science & Technology vs Environment & Climate.
    science_mask = df["iab_category_raw"] == "Science"
    env_mask = science_mask & df["text"].apply(taxonomy.is_environment_article)
    sci_mask = science_mask & ~env_mask

    # Assign labels.
    df["label"] = -1  # placeholder
    df["revised_category"] = ""

    # Direct IAB mappings (non-Science categories).
    for iab_cat in KEEP_IAB:
        if iab_cat == "Science":
            continue
        label = taxonomy.lookup_iab(iab_cat)
        if label is None:
            continue
        mask = df["iab_category_raw"] == iab_cat
        df.loc[mask, "revised_category"] = label
        df.loc[mask, "label"] = taxonomy.label_index(label)

    # Science -> Science & Technology.
    df.loc[sci_mask, "revised_category"] = "Science & Technology"
    df.loc[sci_mask, "label"] = taxonomy.label_index("Science & Technology")

    # Science (environment-filtered) -> Environment & Climate.
    df.loc[env_mask, "revised_category"] = "Environment & Climate"
    df.loc[env_mask, "label"] = taxonomy.label_index("Environment & Climate")

    # Drop any unmapped rows (shouldn't happen).
    df = df[df["label"] >= 0].copy()
    print(f"[filter] final mapped rows: {len(df)}")

    return df


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["revised_category"].value_counts()
    total = len(df)
    out = pd.DataFrame({
        "revised_category": vc.index,
        "count": vc.values,
        "share": (vc.values / total).round(4),
    })
    return out.reset_index(drop=True)


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 80/10/10 split."""
    train_val, test = train_test_split(
        df, test_size=0.1, random_state=config.SPLIT_SEED, stratify=df["label"]
    )
    relative_valid = config.VALID_RATIO / (1 - 0.1)  # 0.1 / 0.9
    train, val = train_test_split(
        train_val, test_size=relative_valid, random_state=config.SPLIT_SEED, stratify=train_val["label"]
    )
    print(f"[split] train={len(train)} val={len(val)} test={len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="smoke-test: limit rows (0 = all)")
    args = ap.parse_args()

    df = load_iab_frame(args.sample)
    df = filter_and_map(df)

    # Write v2 dataset.
    config.GENERATED.mkdir(parents=True, exist_ok=True)
    out_cols = ["text", "revised_category", "label", "iab_category_raw", "url", "domain"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(config.V2_CSV, index=False)
    print(f"[save] wrote {config.V2_CSV}")

    # Write class counts.
    cc = class_counts(df)
    cc.to_csv(config.CLASS_COUNTS_CSV, index=False)
    print(f"[save] wrote {config.CLASS_COUNTS_CSV}")
    print(cc.to_string(index=False))

    # Create and save splits.
    train_df, val_df, test_df = create_splits(df)
    train_df[out_cols].to_csv(config.TRAIN_CSV, index=False)
    val_df[out_cols].to_csv(config.VAL_CSV, index=False)
    test_df[out_cols].to_csv(config.TEST_CSV, index=False)
    print(f"[save] wrote {config.TRAIN_CSV}, {config.VAL_CSV}, {config.TEST_CSV}")

    # Summary.
    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total mapped rows: {len(df)}")
    print(f"Labels: {config.NUM_LABELS}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print()
    print("Class distribution:")
    print(cc.to_string(index=False))


if __name__ == "__main__":
    main()
