"""Shared configuration for the news-domain-tagger v3 build (IAB dataset).

Single source of truth for paths, taxonomy, the <=1 GFLOP operating point,
the Student 4L-256H architecture, seeds, outputs, and device selection.

NOTE: run every module from the PROJECT ROOT so `import config` resolves:
    PYTHONPATH=. python -m train.train_student --help
"""

from pathlib import Path

# ---- Project layout (nothing writes outside generated/) ----
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"
TRAIN_DIR = ROOT / "train"
EVAL_DIR = ROOT / "eval"
DEPLOY_DIR = ROOT / "deploy"
GENERATED = ROOT / "generated"

# ---- Dataset source ----
HF_DATASET = "mdonigian/iab-news-classification"

# ---- Taxonomy: 11 IAB-aligned labels; order == encoding 0..10 ----
LABELS = [
    "Politics & Government",
    "War & Conflicts",
    "Crime & Justice",
    "Law & Legal",
    "Economy & Business",
    "Science & Technology",
    "Health",
    "Education",
    "Disasters & Emergencies",
    "Sports",
    "Environment & Climate",
]
NUM_LABELS = len(LABELS)
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}

# ---- Data split (frozen 80/10/10, seed 42) ----
SPLIT_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1

# ---- Operating point: p99 = 85 tokens; 96 truncates minimal text ----
MAX_LEN = 96

# Gating budget: measured forward FLOP (torch FlopCounterMode, 2xMACs) <= 1e9
GFLOP_BUDGET = 1.0  # student 4L-256H @96 ~= 0.604 GFLOP

# ---- Student architecture (PRIMARY candidate R1: 4L-256H) ----
NUM_LAYERS = 4
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 1024
NUM_ATTENTION_HEADS = 4

# ---- Training constants ----
SEEDS = (42, 1337)         # 2-seed reporting
SEEDS_ARG = "42,1337"      # argparse string form
TEACHER_TOKENIZER = "distilbert-base-uncased"

# ---- Generated datasets ----
V2_CSV = GENERATED / "iab_news_v2.csv"
CLASS_COUNTS_CSV = GENERATED / "class_counts_iab.csv"
TRAIN_CSV = GENERATED / "split_train.csv"
VAL_CSV = GENERATED / "split_val.csv"
TEST_CSV = GENERATED / "split_test.csv"

# ---- Model/checkpoint defaults (Colab-friendly: under a Drive mount) ----
DRIVE_ROOT = Path("/content/drive/MyDrive/ArticleTagging_v3")
OUTPUTS = DRIVE_ROOT / "outputs"

# Teacher checkpoint (train from scratch or distilbert-base-uncased).
TEACHER_OUT = OUTPUTS / "distilbert_teacher_iab"

# Intermediate + student artifacts -> outputs/ (Colab training writes here).
TEACHER_LOGITS = OUTPUTS / "teacher_logits"
STUDENT_OUT = OUTPUTS / "student_4L_256H"
STUDENT_BEST = STUDENT_OUT / "best"
DEPLOYED_STUDENT = OUTPUTS / "deployed_student"
EVAL_REPORT = OUTPUTS / "eval_report"


# ---- Device: CUDA on Colab, CPU fallback for smoke tests ----
def select_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
