# Session Log — 2026-08-18: Building the 10-Label IPTC News Domain Tagger

> **Purpose of this file.** Complete handoff so a fresh session (tomorrow) can resume with zero loss.
> Send this file + the `generated/` folder (or just point to this path) and continue.
> Original `IMPLEMENTATION_PLAN.md` was NOT modified; this is a separate log of the build.
> No pre-existing file was ever overwritten.

---

## 1. Decisions locked in (from user, 2026-08-18)

| Topic | Decision |
|---|---|
| Taxonomy | **10 IPTC-aligned labels.** No "World"/"International"/catch-all. No Society, Conflict, Disaster classes. |
| Labels (order = encoding 0..9) | 0 Politics and Government · 1 Economy, Business and Finance · 2 Crime, Law and Justice · 3 Education · 4 Arts, Culture, Entertainment and Media · 5 Lifestyle and Leisure · 6 Sport · 7 Health · 8 Science and Technology · 9 Environment and Climate |
| Ambiguous rows | Never force labels → `review_ambiguous.csv`, manual review later. |
| Teacher | Retrain the **existing synced 8-label DistilBERT** with new 10-label head + weighted CE + stratified split → save as `distilbert_teacher_revised_10_labels`. Preserve old 8-label checkpoint. Run on Colab T4, not local CPU. |
| Student training | Colab T4 (GPU). Local CPU only for smoke tests. |
| FastAPI app | Do NOT modify yet. Out of build scope for this session. |
| FLOP budget | Gate = **≤1 GFLOP measured as FLOP (torch `FlopCounterMode`, 2×MACs)** ⇒ ≤0.5e9 MACs. Primary candidate passes with margin (0.604 GFLOP = 0.302 GMACs). |
| Stop-point | User requested: display final mapping table + class counts for sign-off **before any training**. (Presented in §4 below; full-size maps/CSVs exist in `generated/`.) |

**Outstanding minor item to confirm:** the user's teacher answer said "revised_11_labels" but the final taxonomy is **10** labels → used `distilbert_teacher_revised_10_labels`. Continue with 10 unless told otherwise.

---

## 2. Environment (verified this machine)

- Working dir: `D:\INTERNSHIP PROJECT\ARTICLE DOMAIN TAGGING\article-domain-tagger`
- Runs with: `venv\Scripts\python.exe` (Python 3.13.5)
- venv packages: torch **2.12.1+cpu**, transformers **5.12.1**, sklearn 1.9.0, pandas 3.0.3, datasets 5.0.0, joblib. **No fastapi/uvicorn/tabulate in venv** (those live in base Anaconda `E:\Anaconda\anaconda3`).
- Node v22.14.0 + npm 11.5.2 available; **no GPU** (`torch.cuda.is_available() == False`).
- HF cache (offline-capable, `HF_HUB_OFFLINE=1`):
  - `C:\Users\Asus\.cache\huggingface\hub\models--distilbert-base-uncased` (cache_path + tokenizer)
  - `C:\Users\Asus\.cache\huggingface\hub\datasets--heegyu--news-category-dataset` (full **209,527** rows → drops resurrected offline)
- Old fine-tuned teacher: **NOT local** — only on Google Drive at `/content/drive/MyDrive/ARTICLE DOMAIN TAGGING /distilbert_news_domain_model`.

---

## 3. Files created today (all NEW, nothing overwritten)

```
config.py                     shared paths, LABELS (10), MAX_LEN=96, SPLIT_SEED=42, budget=1.0 GFLOP
data/taxonomy.py              LabelRule dataclass, 10 label rules (def, include/exclude, confusion pairs),
                              MAPPING (raw category -> revised label), REVIEW_BUCKET (14 categories, hints),
                              LEGACY_MAPPING (old 8-label, used only to preserve final_category)
data/remap_dataset.py         reads full HF cache -> v2 + review + audit + class-count CSVs (Stop-review at end)
data/split_dataset.py         frozen 80/10/10 stratified (seed 42) -> split_train/val/test.csv
analysis/token_stats.py       token length distribution + truncation table (writes generated/token_stats.csv)
analysis/measure_flops.py     patched FlopCounterMode (tabulate missing), per-len FLOP/MACs, CPU latency p50/p95
train/common.py               device, args wiring, inverse-frequency class weights, TextDataset, loaders
train/student.py              Student 4L-256H (BertConfig: 4L, 256H, 1024 inter, 4 heads, 10 labels)
train/teacher_train.py        retrain/head-swap teacher, weighted CE, best-by-val-f1_macro (Colab-ready)
train/cache_teacher_logits.py cache train-split soft targets to NPZ, row-index aligned (Colab-ready)
train/train_student.py        KD: alpha*(T^2)*KL + (1-alpha)*CE, T=3, alpha=0.7, 2 seeds, best-per-seed
eval/evaluate.py              acc, f1_macro, per-class p/r/f, confusion matrix CSV+PNG, optional FLOP re-check
deploy/model_export.py        export with id2label/label2id, reload sanity, FLOP gate re-verify
train/cache_teacher_logits.py & train_student.py etc. — all new; see full list below
```

Full module list (excluding `venv/` and `__init__.py`):

- `config.py`
- `data/remap_dataset.py`, `data/split_dataset.py`, `data/taxonomy.py`
- `analysis/token_stats.py`, `analysis/measure_flops.py`
- `train/teacher_train.py`, `train/cache_teacher_logits.py`, `train/train_student.py`, `train/common.py`, `train/student.py`
- `eval/evaluate.py`
- `deploy/model_export.py`

---

## 4. Generated artifacts (in `generated/`)

| File | Rows | Notes |
|---|---|---|
| `news_domain_tagging_v2.csv` | 167,019 | original 7 cols + `text` + `revised_category` + `label` (0..9) |
| `review_ambiguous.csv` | 42,508 | ambiguous rows + `keyword_hint` (manual-review later) |
| `mapping_audit.csv` | 42 | per raw category: count, legacy_label, revised_label, decision, hint |
| `class_counts_revised.csv` | 10 | counts + share per revised label |
| `split_train.csv` | 133,615 | frozen |
| `split_val.csv` | 16,701 | frozen |
| `split_test.csv` | 16,703 | frozen |
| `token_stats.csv` | — | summary + truncation rows |

**Final class counts (STOP-REVIEW table 1 — validated: 167,019 mapped + 42,508 review = 209,527):**

| Label | Count | Share |
|---|---|---|
| Lifestyle and Leisure | 50,229 | 30.1% |
| Politics and Government | 35,602 | 21.3% |
| Arts, Culture, Entertainment and Media | 29,628 | 17.7% |
| Health | 24,639 | 14.8% |
| Economy, Business and Finance | 7,748 | 4.6% |
| Sport | 5,077 | 3.0% |
| Science and Technology | 4,310 | 2.6% |
| Environment and Climate | 4,066 | 2.4% |
| Crime, Law and Justice | 3,562 | 2.1% |
| Education | 2,158 | 1.3% |

**Review bucket (14 raw categories, `keyword_hint` in CSV):** LATINO VOICES, BLACK VOICES, QUEER VOICES, WOMEN, IMPACT, WEIRD NEWS, GOOD NEWS, RELIGION, U.S. NEWS, WORLD NEWS, THE WORLDPOST, WORLDPOST, HOME & LIVING, FIFTY. (HOME & LIVING drifted to streaming/entertainment per sampled headlines.)

Notable re-maps vs legacy (flagged for review): WELLNESS (Lifestyle→**Health**), MEDIA + GREEN (Business→**Arts/Media** and **Environment**).

---

## 5. Measurements taken (matched IMPLEMENTATION_PLAN.md)

**Token stats (v2 corpus, 167,019 texts, distilbert tokenizer):**
p50=39 · p75=49 · p90=62 · p95=71 · p99=85 · max=353.
Truncation if max_len: 32→67.6%, 48→25.7%, 64→8.6%, **96→0.28%**, 128→0.10%. Operating point **96**.

**FLOP measurements (torch convention, FLOP = 2×MACs):**

| Model | @96 | @128 | Params |
|---|---|---|---|
| Student 4L-256H | **0.604 GFLOP (0.302 GMACs)** ✓ gate | 0.805 | 11,173,130 (44.7 MB fp32) |
| distilbert-base-uncased (ref) | 8.155 | 10.873 | 66,961,162 |

CPU latency (student @96, local): p50 ≈ 5.8 ms, p95 ≈ 7.3 ms (200 fwd).

---

## 6. Smoke tests passed (local CPU, tiny samples)

Full chain validated end-to-end; all artifacts removed after test:

1. `python -m train.teacher_train --teacher-from distilbert-base-uncased --out generated/teacher_smoke --sample 200 --epochs 1 --batch-size 8 --workers 0` → saved checkpoint ✓
2. `python -m train.cache_teacher_logits --teacher generated/teacher_smoke --sample 200 --batch-size 8 --workers 0 --out generated/teacher_logits_smoke` → 200×10 logits, row-aligned ✓
3. `python -m train.train_student --logits generated/teacher_logits_smoke/train_logits.npz --sample 200 --epochs 1 --batch-size 8 --workers 0 --out generated/student_smoke --seeds 42` → loss 0.695 (CE 2.30→2.30, KL→0.0007) ✓
4. `python -m eval.evaluate --model generated/student_smoke_seed42 --out generated/eval_smoke --flops` → report + PNG + FLOP re-check **PASS 0.604** ✓
5. `python -m deploy.model_export --model generated/student_smoke_seed42 --out generated/deployed_student --verify-flops` → 11.17M params, gate PASS, reload sanity ✓

Bugs found & fixed during the build: `batch_size` kwarg not supported in tokenizer call; NaN round-trip in `text` (fillna fix); `FlopCounterMode.get_table` missing tabulate → patch; `TextDataset.__getitem__` needed tensor conversion; student `step` counter reset per epoch; missing `out_dir` arg in run_training call; syntax error in measure_flops helper.

**Note on smoke accuracy figures:** tiny-sample runs (200 rows / 1 epoch) deliberately produce near-random f1 — **they are pipeline-proving runs, not quality signals.** Real training runs full data on Colab.

---

## 7. Resume steps (tomorrow / Colab T4)

Workflow is **data-prep local (done) → training Colab → download → evaluate/export local**.

On Colab (GPU T4):
1. Mount Drive (`from google.colab import drive; drive.mount('/content/drive')`), upload `generated/` (splits + CSVs) and the package files, or clone this folder.
2. **Train teacher (10 labels):**
   `python -m train.teacher_train --teacher-from "/content/drive/MyDrive/ARTICLE DOMAIN TAGGING /distilbert_news_domain_model" --data-dir generated --out "/content/drive/MyDrive/ARTICLE DOMAIN TAGGING /distilbert_teacher_revised_10_labels" --device cuda --batch-size 32 --epochs 3`
   (fallback teacher-source if old ckpt missing: `distilbert-base-uncased`)
3. **Cache logits:**
   `python -m train.cache_teacher_logits --teacher "<revised teacher path>" --data-dir generated --out generated/teacher_logits --fp16`
4. **Train student (KD, 2 seeds):**
   `python -m train.train_student --logits generated/teacher_logits/train_logits.npz --data-dir generated --out generated/student_4L_256H --device cuda --batch-size 64 --epochs 3 --seeds 42,1337 --temp 3 --alpha 0.7`
5. Download best seed dir (`generated/student_4L_256H_seed42`, `_seed1337`) back to local `generated/`.

On local CPU:
6. Evaluate: `python -m eval.evaluate --model generated/student_4L_256H_seed42 --flops`
7. Export: `python -m deploy.model_export --model generated/student_4L_256H_seed42 --out generated/deployed_student --verify-flops`
8. Optionally open `generated/review_ambiguous.csv` for the manual-review pass later (feeds more + cleaner data for Education/Crime).

**Expected quality target (from IMPLEMENTATION_PLAN.md):** gate on val/test macro-F1 ≈ ≥0.80 ±0.02 (teacher ceiling without truncation @128), watch Education (2,158) and Crime (3,562) per-class F1. Realistic path: teacher on 10 labels will be ≥ baseline; student distills from it.

**Fallbacks (plan §E):** budget fail → max_len 128→96→64 (student 0.81→0.60→0.40 GFLOP). Quality fail → promote to TinyBERT 4L-312D @96 (0.87 GFLOP) with same KD; or add hidden-state alignment; R6 2L-256H only if CPU latency unacceptable.

---

## 8. Commands that worked (local, quick reference)

```powershell
# Data prep (already run; re-runnable/auditable)
$env:HF_HUB_OFFLINE="1"; venv\Scripts\python.exe -m data.remap_dataset --source heegyu
venv\Scripts\python.exe -m data.split_dataset

# Analysis
venv\Scripts\python.exe -m analysis.token_stats
venv\Scripts\python.exe -m analysis.measure_flops --arch student --lengths 32,64,96,128
venv\Scripts\python.exe -m analysis.measure_flops --arch teacher --lengths 96,128
```

---

## 9. Known open items / next actions

- [ ] Confirm 10-label teacher folder name vs "11_labels" wording (use `distilbert_teacher_revised_10_labels`).
- [ ] Sign-off the STOP-REVIEW mapping (all 10 labels + review bucket) — full tables in `generated/`.
- [ ] Manual review of the 42,508 ambiguous rows (later; not blocking training).
- [ ] Colab: teacher → logits → student; download; evaluate; export.
- [ ] Later (out of scope this session): FastAPI app (HF `AutoModelForSequenceClassification` path), React frontend, optional ONNX/INT8 latency pass, deployed-student sync.