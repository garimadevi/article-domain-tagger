# Article Domain Tagger v3 — Colab End-to-End Training Script
#
# Run this in Google Colab with T4 GPU runtime.
# Steps:
#   1. Mount Google Drive
#   2. Clone/upload the project
#   3. Run this script section by section
#
# Total time: ~2-2.5 hours

# ============================================================
# SECTION 0: Setup
# ============================================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone or upload the project
# Option A: Upload via Colab file browser
# Option B: Clone from GitHub
# !git clone <your-repo-url> /content/article-domain-tagger

# Set working directory
import os
os.chdir('/content/article-domain-tagger')

# Install dependencies
!pip install -q transformers datasets torch scikit-learn pandas numpy

# Verify setup
import config
print(f"Labels ({config.NUM_LABELS}): {config.LABELS}")
print(f"Student: {config.NUM_LAYERS}L-{config.HIDDEN_SIZE}H")

# ============================================================
# SECTION 1: Load and Prepare Dataset
# ============================================================

# This downloads ~106K articles from HuggingFace, filters to 11 political
# categories, extracts Environment articles from Science via keywords,
# and creates 80/10/10 stratified splits.
!PYTHONPATH=. python -m data.load_iab_dataset

# Check the generated files
import pandas as pd
print("\n=== Class Distribution ===")
cc = pd.read_csv(config.CLASS_COUNTS_CSV)
print(cc.to_string(index=False))

print(f"\n=== Splits ===")
train = pd.read_csv(config.TRAIN_CSV)
val = pd.read_csv(config.VAL_CSV)
test = pd.read_csv(config.TEST_CSV)
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# ============================================================
# SECTION 2: Train DistilBERT Teacher (11 labels)
# ============================================================

# Train from distilbert-base-uncased (fresh, no old checkpoint needed)
!PYTHONPATH=. python -m train.teacher_train \
    --teacher-from distilbert-base-uncased \
    --out "/content/drive/MyDrive/ArticleTagging_v3/outputs/distilbert_teacher_iab" \
    --device cuda --max-len 96 --batch-size 32 --epochs 3 --lr 2e-5 \
    --seed 42 --weights-scheme inverse --save-every-epoch

# ============================================================
# SECTION 3: Cache Teacher Logits
# ============================================================

!PYTHONPATH=. python -m train.cache_teacher_logits \
    --teacher "/content/drive/MyDrive/ArticleTagging_v3/outputs/distilbert_teacher_iab" \
    --data-dir generated \
    --out "/content/drive/MyDrive/ArticleTagging_v3/outputs/teacher_logits" \
    --device cuda --max-len 96 --batch-size 64 --fp16

# ============================================================
# SECTION 4: Train Student (4L-256H, Knowledge Distillation)
# ============================================================

# Train with 2 seeds for variance reporting
!PYTHONPATH=. python -m train.train_student \
    --logits "/content/drive/MyDrive/ArticleTagging_v3/outputs/teacher_logits/train_logits.npz" \
    --out "/content/drive/MyDrive/ArticleTagging_v3/outputs/student_4L_256H" \
    --device cuda --max-len 96 --batch-size 64 --epochs 5 --lr 5e-4 \
    --temp 3.0 --alpha 0.7 --seeds "42,1337"

# ============================================================
# SECTION 5: Evaluate on Test Set
# ============================================================

# Evaluate the best student model (seed 42)
!PYTHONPATH=. python -m eval.evaluate \
    --model "/content/drive/MyDrive/ArticleTagging_v3/outputs/student_4L_256H_seed42" \
    --data-dir generated \
    --out "/content/drive/MyDrive/ArticleTagging_v3/outputs/eval_report" \
    --batch-size 64 --flops

# Print results
import pandas as pd
report = pd.read_csv("/content/drive/MyDrive/ArticleTagging_v3/outputs/eval_report/per_class.csv")
print("\n=== Test Set Results ===")
print(report.to_string(index=False))

# ============================================================
# SECTION 6: Quick Sanity Test
# ============================================================

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "/content/drive/MyDrive/ArticleTagging_v3/outputs/student_4L_256H_seed42"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Test with sample texts
test_texts = [
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
]

with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=96, padding=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
        conf = probs[0][pred].item()
        print(f"[{conf:.2%}] {config.ID2LABEL[pred]:.<30} | {text[:60]}...")
