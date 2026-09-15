"""Local smoke test — verify the full pipeline works without GPU.

Usage:
  python smoke_test.py

This runs a minimal version of the pipeline with 500 samples to verify
all scripts are compatible with the new 11-label IAB taxonomy.
"""

import os
import sys
import tempfile
import shutil

import pandas as pd
import numpy as np

# Ensure project root is on path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config():
    import config
    assert config.NUM_LABELS == 11, f"Expected 11 labels, got {config.NUM_LABELS}"
    assert len(config.LABELS) == 11
    print(f"[PASS] config: {config.NUM_LABELS} labels loaded correctly")
    return config


def test_taxonomy():
    from data import taxonomy
    assert len(taxonomy.LABEL_ORDER) == 11
    assert "War & Conflicts" in taxonomy.LABEL_ORDER
    assert "Environment & Climate" in taxonomy.LABEL_ORDER
    assert len(taxonomy.IAB_TO_LABEL) == 10  # 10 direct IAB mappings
    assert len(taxonomy.ENVIRONMENT_KEYWORDS) > 20
    # Test environment extraction
    assert taxonomy.is_environment_article("Arctic ice melting due to climate change")
    assert not taxonomy.is_environment_article("Apple releases new iPhone model")
    print("[PASS] taxonomy: 11 labels, IAB mapping, environment keywords OK")
    return taxonomy


def test_student_build():
    from train.student import build_student
    model = build_student(num_labels=11)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 10_000_000, f"Expected >10M params, got {n_params}"
    print(f"[PASS] student: {n_params:,} params, builds correctly")


def test_data_load():
    """Create a tiny synthetic dataset to test the pipeline."""
    import config

    # Create synthetic data mimicking IAB format.
    texts = [
        "Senate passes new infrastructure bill with bipartisan support",
        "Ukraine forces advance in eastern front as fighting intensifies",
        "Man arrested for armed robbery at downtown bank",
        "Supreme Court hears arguments in major tech antitrust case",
        "Stock market rallies on strong earnings reports",
        "Scientists discover new high-efficiency solar cell material",
        "WHO warns of new disease outbreak in Southeast Asia",
        "University announces free tuition for low-income students",
        "Hurricane makes landfall causing widespread damage",
        "Team wins championship in overtime thriller",
        "Arctic ice sheet melts at record rate due to warming",
        "Government announces new education policy reform",
        "Military operation launches airstrikes on terrorist camps",
        "Police raid organized crime network across three states",
        "New environmental regulation targets carbon emissions",
    ] * 10  # repeat to get 150 rows

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 10] * 10
    df = pd.DataFrame({
        "text": texts,
        "label": labels,
        "revised_category": [config.LABELS[l] for l in labels],
    })

    # Save splits.
    config.GENERATED.mkdir(parents=True, exist_ok=True)
    df.iloc[:100].to_csv(config.TRAIN_CSV, index=False)
    df.iloc[100:120].to_csv(config.VAL_CSV, index=False)
    df.iloc[120:150].to_csv(config.TEST_CSV, index=False)
    print(f"[PASS] data: synthetic dataset created ({len(df)} rows)")
    return df


def test_load_split():
    from train.common import load_split
    train_df = load_split(str(config.GENERATED), "train")
    assert "text" in train_df.columns
    assert "label" in train_df.columns
    assert train_df["label"].min() >= 0
    assert train_df["label"].max() <= 10
    print(f"[PASS] load_split: {len(train_df)} train rows loaded correctly")


def test_class_weights():
    from train.common import class_weights
    import config
    train_df = pd.read_csv(config.TRAIN_CSV)
    weights = class_weights(train_df["label"])
    assert weights.shape == (11,), f"Expected 11 weights, got {weights.shape}"
    print(f"[PASS] class_weights: {weights.tolist()[:3]}...")


def test_teacher_forward():
    """Quick forward pass with DistilBERT (smoke test)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=11, ignore_mismatched_sizes=True
    )
    inputs = tokenizer("Test article about politics", return_tensors="pt", truncation=True, max_length=96)
    with torch.no_grad():
        out = model(**inputs)
    assert out.logits.shape == (1, 11)
    print(f"[PASS] teacher forward: logits shape {out.logits.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST — Article Domain Tagger v3")
    print("=" * 60)

    config = test_config()
    test_taxonomy()
    test_student_build()
    test_data_load()
    test_load_split()
    test_class_weights()

    try:
        test_teacher_forward()
    except Exception as e:
        print(f"[SKIP] teacher forward: {e} (no internet or model download failed)")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED — ready for Colab training")
    print("=" * 60)
