"""Shared helpers for teacher/logits/student training scripts.

All training scripts are designed to run on Colab T4 (GPU) with a mounted
Drive, but fall back gracefully to local CPU for smoke tests.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


def select_device() -> torch.device:
    return torch.device(config.select_device())


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-dir", default=str(config.GENERATED))
    parser.add_argument("--max-len", type=int, default=config.MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="auto=cuda if available")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sample", type=int, default=0, help="smoke-test: limit rows (0 = all)")
    return parser


def device_from_args(args) -> torch.device:
    if args.device:
        return torch.device(args.device)
    return select_device()


def class_weights(labels: pd.Series, scheme: str = "inverse") -> torch.Tensor:
    """Inverse-frequency weights w_c = N/(K*N_c)."""
    counts = labels.value_counts().sort_index()
    k = len(counts)
    n = int(counts.sum())
    w = np.array([n / (k * counts.get(i, 1)) for i in range(k)], dtype=np.float32)
    return torch.tensor(w)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_len
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "label": self.labels[idx],
        }


def load_split(data_dir, name: str, sample: int = 0) -> pd.DataFrame:
    csv = {
        "train": "split_train.csv",
        "val": "split_val.csv",
        "test": "split_test.csv",
    }[name]
    df = pd.read_csv(f"{data_dir}/{csv}")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    if sample:
        df = df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)
    return df


def make_loaders(df, tokenizer, max_len, batch_size, workers, shuffle, device):
    ds = TextDataset(df["text"], df["label"].values, tokenizer, max_len)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=device == "cuda"
    )
    return dl