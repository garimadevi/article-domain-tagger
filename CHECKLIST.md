# Article Domain Tagger — Master Checklist (v3: IAB Dataset)

> Legend: `[x]` done (verified) · `[ ]` pending/not started
> Last updated: 2026-09-07

**Core-pipeline progress (Phases A–E): 5 / 15 items — ~33%**

---

## Phase A — Dataset & Taxonomy (`done`)

- [x] 11-label IAB taxonomy locked (`config.py`, `data/taxonomy.py`)
- [x] IAB dataset loaded from HuggingFace (106K articles, 92 sources)
- [x] Environment articles extracted from Science via keyword filtering
- [x] Frozen 80/10/10 stratified splits (seed 42)

## Phase B — Analysis (`pending`)

- [ ] Token stats, operating point max-len 96
- [ ] Student FLOP ≤ 1 GFLOP gate PASS
- [ ] CPU latency estimate

## Phase C — Pipeline code + smoke tests (`done`)

- [x] `train/teacher_train.py`, `train/cache_teacher_logits.py`, `train/train_student.py`, `train/common.py`, `train/student.py`
- [x] `eval/evaluate.py`, `data/*`
- [x] Config updated for 11 labels
- [x] Colab training script created

## Phase D — Teacher training (11 labels) (`pending`)

- [ ] Train DistilBERT teacher on IAB dataset
- [ ] Save best validation checkpoint
- [ ] Confirm best epoch

## Phase E — Student training + eval (`pending`)

- [ ] Cache teacher logits (KD stage)
- [ ] Train student 4L-256H with KD
- [ ] Evaluate on test set
- [ ] Prepare final comparison/report

## Phase F — Deferred (out of current scope)

- [ ] Manual label verification / noise reduction
- [ ] FastAPI app (HF `AutoModelForSequenceClassification` path)
- [ ] React frontend
- [ ] Optional ONNX/INT8 latency pass

---

## Key paths

| Item | Path |
|---|---|
| Dataset source | `mdonigian/iab-news-classification` (HuggingFace) |
| Generated data | `generated/` (local) |
| Colab outputs | `/content/drive/MyDrive/ArticleTagging_v3/outputs/` |
| Training script | `colab_train.py` (end-to-end) |
