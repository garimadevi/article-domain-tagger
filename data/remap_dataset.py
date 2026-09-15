"""Remap the full HuffPost dataset to the IPTC-aligned taxonomy v2.

Never overwrites anything. Writes, under generated/:
  - news_domain_tagging_v2.csv  (mapped rows: original cols + text + revised_category + label)
  - review_ambiguous.csv        (rows with no safe auto-mapping, with keyword hint)
  - mapping_audit.csv           (per raw category: count, legacy label, revised label, decision)
  - class_counts_revised.csv    (per revised label: count + share)

Usage:
  python -m data.remap_dataset [--source heegyu|csv]
     --source heegyu : read the full 209,527-row HF cache (default; resurrects dropped rows)
     --source csv    : read only news_domain_tagging_cleaned.csv (188,795 rows, legacy subset)

At the end of the run it prints STOP-review tables (mapping + class counts) for sign-off.
"""

import argparse
import sys

import pandas as pd

import config
from data import taxonomy


def load_frame(source: str) -> pd.DataFrame:
    if source == "heegyu":
        from datasets import load_dataset
        ds = load_dataset(config.HF_DATASET, split="train")
        df = pd.DataFrame(ds)
    else:
        df = pd.read_csv(config.SOURCE_CSV)
    return df


def make_text(df: pd.DataFrame) -> pd.Series:
    return (df["headline"].fillna("") + " " + df["short_description"].fillna("")).str.strip()


def build(df: pd.DataFrame):
    df = df.copy()

    # Preserve originals: recompute legacy final_category so v2 carries it even
    # for rows the old mapping never merged (they were dropped, now resurrected).
    df["text"] = make_text(df)
    df["final_category"] = df["category"].map(taxonomy.LEGACY_MAPPING)

    # New deterministic mapping; unmatched -> review.
    df["revised_category"] = df["category"].map(taxonomy.MAPPING)
    df["label"] = df["revised_category"].map(
        {lbl: i for i, lbl in enumerate(taxonomy.LABEL_ORDER)}
    )

    review_mask = df["revised_category"].isna()
    review_df = df[review_mask].copy()
    review_df["keyword_hint"] = review_df["category"].map(taxonomy.REVIEW_BUCKET)
    mapped_df = df[~review_mask].copy()

    # cols: keep original 7 + text + revised_category + label
    keep_cols = ["link", "headline", "category", "short_description", "authors", "date",
                 "final_category", "text", "revised_category", "label"]
    mapped_df = mapped_df[keep_cols].sort_values("link").reset_index(drop=True)

    review_out = ["link", "headline", "category", "short_description", "authors", "date",
                  "final_category", "text", "keyword_hint"]
    review_out = [c for c in review_out if c in review_df.columns]
    review_df = review_df[review_out].sort_values("link").reset_index(drop=True)

    return mapped_df, review_df


def audit_rows(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, grp in df_all.groupby("category", sort=False):
        rows.append({
            "category": cat,
            "count": len(grp),
            "legacy_label": taxonomy.LEGACY_MAPPING.get(cat, "NaN"),
            "revised_label": taxonomy.MAPPING.get(cat, "REVIEW"),
            "decision": "mapped" if taxonomy.is_review(cat) is False and taxonomy.lookup(cat) is not None else "review",
            "keyword_hint": taxonomy.REVIEW_BUCKET.get(cat, ""),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def class_counts(mapped_df: pd.DataFrame) -> pd.DataFrame:
    vc = mapped_df["revised_category"].value_counts()
    total = len(mapped_df)
    out = pd.DataFrame({
        "revised_category": vc.index,
        "count": vc.values,
        "share": (vc.values / total).round(4),
    })
    return out.reset_index(drop=True)


def print_stop_review(mapped_df: pd.DataFrame, review_df: pd.DataFrame, all_df: pd.DataFrame):
    sep = "=" * 78
    print(sep)
    print("STOP-REVIEW TABLE 1 — FINAL TAXONOMY MAP (10 labels)")
    print(sep)
    cc = class_counts(mapped_df)
    print(cc.to_string(index=False))
    print()
    print(sep)
    print("STOP-REVIEW TABLE 2 — MAPPING AUDIT (decisions per raw category)")
    print(sep)
    print(audit_rows(all_df).to_string(index=False))
    print()
    print(f"mapped rows : {len(mapped_df)}")
    print(f"review rows : {len(review_df)}")
    print(f"total input : {len(all_df)}")
    print()
    print("REVIEW BUCKET (no row auto-assigned) — keyword hints included in review_ambiguous.csv")
    print(review_df[["category", "keyword_hint"]].drop_duplicates("category").to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["heegyu", "csv"], default="heegyu")
    args = ap.parse_args()

    df = load_frame(args.source)
    print(f"[remap] loaded {len(df)} rows from source={args.source}")

    # sanity: every raw category must be covered by mapping OR review bucket
    known = set(taxonomy.MAPPING) | set(taxonomy.REVIEW_BUCKET)
    unknown = sorted(set(df["category"].dropna().unique()) - known)
    if unknown:
        print(f"[remap] WARNING: {len(unknown)} categories not in taxonomy/review: {unknown}")
        sys.exit(1)

    mapped_df, review_df = build(df)

    config.GENERATED.mkdir(parents=True, exist_ok=True)
    mapped_df.to_csv(config.V2_CSV, index=False)
    review_df.to_csv(config.REVIEW_CSV, index=False)
    audit_rows(df).to_csv(config.MAPPING_AUDIT_CSV, index=False)
    class_counts(mapped_df).to_csv(config.CLASS_COUNTS_CSV, index=False)

    print(f"[remap] wrote:")
    print(f"  {config.V2_CSV}")
    print(f"  {config.REVIEW_CSV}")
    print(f"  {config.MAPPING_AUDIT_CSV}")
    print(f"  {config.CLASS_COUNTS_CSV}")
    print()
    print_stop_review(mapped_df, review_df, df)


if __name__ == "__main__":
    main()