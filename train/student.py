"""Student model builder shared by measure_flops / train_student / export.

Primary candidate: Student 4L-256H — a small BERT-architecture
(BertForSequenceClassification) with 4 layers, hidden 256, inter 1024,
4 heads, 10 labels, reusing the distilbert (BERT-uncased) vocab.
Measured forward @seq96 ~= 0.60 GFLOP (2xMACs) == 0.30 GMACs.
"""

from transformers import BertConfig, BertForSequenceClassification

import config

VOCAB_SIZE = 30522


def student_config(
    num_labels: int = config.NUM_LABELS,
    layers: int = config.NUM_LAYERS,
    hidden: int = config.HIDDEN_SIZE,
    inter: int = config.INTERMEDIATE_SIZE,
    heads: int = config.NUM_ATTENTION_HEADS,
    max_len: int = config.MAX_LEN,
) -> BertConfig:
    return BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=inter,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
        num_labels=num_labels,
    )


def build_student(num_labels: int = config.NUM_LABELS, seed: int | None = None) -> BertForSequenceClassification:
    if seed is not None:
        import torch
        torch.manual_seed(seed)
    return BertForSequenceClassification(student_config(num_labels=num_labels))