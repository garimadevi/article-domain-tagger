# Implementation Plan — 8-Label News Domain Tagger: Taxonomy Redesign + ≤1 GFLOP Inference

> Status: approved-for-planning, not yet implemented. Created 2026-08-17.
> Resume point: begin with Section E "Recommended Path" → step 1 (taxonomy v2 remap).

Scope note: The repo described (React frontend, FastAPI backend, local HF model checkpoint) is **not present in the workspace**. This directory currently contains only:
- `article-domain-tagger/01_data_loading.ipynb`
- `article-domain-tagger/news_domain_tagging_cleaned.csv` (188,795 rows, 61 MB)
- `article-domain-tagger/venv/`
- `D:\INTERNSHIP PROJECT\Copy of Untitled0.ipynb` (the DistilBERT training notebook, Colab, in the parent folder)

Everything below assumes those missing parts (backend, frontend, checkpoint) get located or attached; flags listed in section E.

---

## A. Current-State Audit (findings on this machine)

**Dataset.** `news_domain_tagging_cleaned.csv` — 7 columns (`link, headline, category, short_description, authors, date, final_category`). Combined text = `headline + " " + short_description`, stripped (`01_data_loading.ipynb` cell 7). Class counts (already imbalanced, 27×):

| Label | Count |
|---|---|
| Lifestyle | 58,330 |
| Politics | 54,718 |
| Entertainment | 37,413 |
| Business | 15,418 |
| World | 12,119 |
| Sports | 5,077 |
| Crime | 3,562 |
| Education | 2,158 |

20,732 rows were dropped as `NaN` (raw categories not in the old mapping: e.g., `SCIENCE, ENVIRONMENT, RELIGION, U.S. NEWS, WEIRD NEWS, GOOD NEWS`, etc. — all rejected).

**Raw `category` (42 HuffPost sections) is still present in the CSV.** This matters: the taxonomy can be re-derived from the raw column, not from the already-merged `final_category`.

**Old label mapping** (`01_data_loading.ipynb` cells 3–4): e.g. `POLITICS/IMPACT/BLACK VOICES/QUEER VOICES/LATINO VOICES/WOMEN → Politics`; `BUSINESS/MONEY/TECH/MEDIA/GREEN → Business`; `ENTERTAINMENT/COMEDY/ARTS/CULTURE & ARTS/STYLE & BEAUTY/STYLE → Entertainment`; `WELLNESS/TRAVEL/FOOD & DRINK/HOME & LIVING/PARENTING/PARENTS/WEDDINGS/DIVORCE → Lifestyle`; `WORLD NEWS/THE WORLDPOST/WORLDPOST/RELIGION → World`; `CRIME → Crime`; `EDUCATION/COLLEGE → Education`; `SPORTS → Sports`.

**Preprocessing:** none beyond concatenation + strip + `LabelEncoder` (alphabetical: Business=0 … World=7) in the training notebook. No lower-casing/text cleaning (DistilBERT tokenizer handles case).

**Training config** (`Copy of Untitled0.ipynb`): distilbert-base-uncased, `max_length=128`, truncate+pad, LR 2e-5, batch 16, 3 epochs, `save_strategy/eval_strategy="epoch"`, `load_best_model_at_end` on `f1_macro`, 80/20 stratified split (val = 37,759 rows).

**Weighted loss: NO.** Neither notebook passes class weights / `pos_weight` / focal loss. The transformer uses default unweighted `CrossEntropyLoss`.

**Checkpoint:** saved only to Google Drive at `/content/drive/MyDrive/ARTICLE DOMAIN TAGGING /distilbert_news_domain_model` — **not on this machine**, no `.safetensors`/`.bin` present locally. The 83.8% acc / 0.80 macro-F1 figures cannot be reproduced locally without it.

**Token-length distribution** (actual DistilBERT subword tokenizer, offline, cached weights):

| p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| 39 | 49 | 62 | 71 | 85 | 353 |

Truncation impact: `max_len=32` → 67.8% truncated; `48` → 25.4%; `64` → 8.6%; **`96` → 0.32%**; `128` → 0.11%. So **max_length=128 is far beyond what the data needs; 96 covers 99.7%**.

**Current GFLOPs, measured** (reproducible method below): DistilBERT, bs=1, seq=128, fp32 = **10.87 GFLOP** — **10.9× over the 1 GFLOP target**. Also measured at seq 96 = 8.15 GFLOP.

Reproducible measurement method (used for all numbers in this plan):
- Python 3.13, `torch 2.12.1+cpu`, `transformers 5.12.1`; `torch.utils.flop_counter.FlopCounterMode`.
- Patch `FlopCounterMode.get_table = lambda *a, **k: ""` (the installed `tabulate` is missing; without the patch the context-manager's `__exit__` raises — this is why naive usage fails).
- `model.eval()`; feed `input_ids=ones(1,L)`, `attention_mask=ones(1,L)`; inside the context run forward under `torch.no_grad()`; read cumulative `get_total_flops()` after each length; take consecutive differences. GFLOP = FLOP/1e9, where FLOP counts multiply–adds ×2 (torch convention).

---

## B. Taxonomy Redesign

Objective: kill the vague "World", keep 8 output labels (minimize MLP/deployment churn), every label gets a one-sentence include/exclude rule, and remapping is reproducible + never overwrites originals.

**Proposed 8 categories:**

1. **Business** — *Include:* companies, markets, finance, the economy, jobs/labor economics, and tech/media industry coverage. *Exclude:* government fiscal/tax-policy debate (→ Politics) and consumer health/wellness products (→ Lifestyle).
2. **Politics** — *Include:* elections, government, legislation, policy decisions, political figures/parties, and US-angled foreign policy. *Exclude:* pure market/earnings news (→ Business) and non-US general news reported for the event itself (→ International News).
3. **Crime** — *Include:* criminal acts, law enforcement, courts, sentencing, investigations, prosecutorial actions. *Exclude:* civil lawsuits without a criminal element (→ Business/Politics by topic) and fictional crime in entertainment (→ Entertainment).
4. **Education** — *Include:* K-12/universities, teachers, curricula, student life, admissions, education policy. *Exclude:* consumer self-improvement/learning content (→ Lifestyle).
5. **Entertainment** — *Include:* film/TV/music/streaming, celebrities, arts, culture, comedy, awards, celebrity gossip. *Exclude:* entertainment-industry M&A/earnings (→ Business) and performer wellness/personal content (→ Lifestyle).
6. **Lifestyle** — *Include:* personal health/wellness, food & cooking, travel & tourism how-to, home & family, parenting, relationships/weddings, consumer fashion, religion-as-practice features. *Exclude:* healthcare/lifestyle-industry news (→ Business) and travel advisories/policy (→ Politics or International News).
7. **Sports** — *Include:* competitive events, teams, athletes, leagues, results/standings. *Exclude:* sports business/finance (→ Business), athlete lifestyle profiles (→ Lifestyle).
8. **International News** (replaces "World") — *Include:* stories whose primary event/actor is located outside the US **and** whose topic is general news (foreign politics, disasters, diplomacy, foreign-leader profiles). *Exclude:* US foreign-policy framing (→ Politics), foreign markets (→ Business), foreign sports (→ Sports), foreign crime (→ Crime), travel guides (→ Lifestyle). *Mutual-exclusivity rule: topic beats geography — the thematic buckets (Business/Crime/Sports/Entertainment/Lifestyle/Politics) absorb their geographies; International hosts only general-news non-US events.*

**Reproducible remapping.** Re-derive from the raw `category` column (kept in the CSV), via a *constant lookup table* — no heuristics in the mixing step:

- Deterministic re-map: `BUSINESS, MONEY, TECH, MEDIA → Business`; `POLITICS → Politics`; `CRIME → Crime`; `EDUCATION, COLLEGE → Education`; `ENTERTAINMENT, COMEDY, ARTS, CULTURE & ARTS → Entertainment`; `SPORTS → Sports`; `WELLNESS, HEALTHY LIVING, FOOD & DRINK, TASTE, TRAVEL, HOME & LIVING, PARENTING, PARENTS, WEDDINGS, DIVORCE, FIFTY → Lifestyle`; `WORLD NEWS → International News`.
- **Content-split or manual-review bucket (cannot be safely auto-mapped):** `IMPACT` (social-issues → Politics vs Society), `GREEN` (climate → Politics/Business/Lifestyle), `THE WORLDPOST` / `WORLDPOST` (opinion/global hybrid → International vs Politics), `U.S. NEWS` (national mixed → Politics vs Education vs …), `WEIRD NEWS` (oddity → Crime/Entertainment/Lifestyle), `GOOD NEWS` (human-interest → Lifestyle/Entertainment), `RELIGION` (→ Lifestyle vs new "Religion" slot if added), `SCIENCE` / `ENVIRONMENT` (→ new Science slot if added), `BLACK VOICES` / `QUEER VOICES` / `LATINO VOICES` / `WOMEN` (identity voice sections = genuinely mixed topics — *previously dumped wholesale into Politics, which is exactly the imprecision the mentor flagged*), and `STYLE & BEAUTY` / `STYLE` (Entertainment vs Lifestyle — a design decision).
- The review bucket is written to `review_ambiguous.csv` with raw category, headline, description, and a suggested-keyword hint. **No row is silently assigned.** Counts per raw category are exported to a mapping report so you can audit every decision.
- **Nothing overwrites the source.** New outputs are `news_domain_tagging_v2.csv` (mapped rows) + `review_ambiguous.csv`; `news_domain_tagging_cleaned.csv` is untouched. Optionally, you ear-mark a **manual decision file** mapping each ambiguous raw category → final label after your 15-minute review.

By default ~99%+ of the 20,732 previously-dropped rows become *usable* → more data for the two worst classes (Education 2,158, Crime 3,562), which is exactly what macro F1 needs.

---

## C. Model Path Comparison (≤1 GFLOP)

Verified FLOP table — bs=1, fp32 forward, **measured with the method above**, FLOP = 2×MACs. GFLOP = FLOP/1e9.

| Architecture | Params | seq32 | seq48 | seq64 | seq96 | seq128 | ≤1 GFLOP? |
|---|---|---|---|---|---|---|---|
| DistilBERT base (current) | 67.0M | 2.72 | 4.08 | 5.44 | 8.16 | **10.87** | ✗ at every seq |
| ALBERT-base (12L-768H, shared) | 11.7M | 5.44 | 8.17 | 10.89 | 16.33 | 21.77 | ✗ at every seq |
| MobileBERT (24L-128H) | 19.5M | 0.99 | 1.48 | 1.97 | 2.95 | 3.94 | ✗ except seq≤32 |
| MiniLM-L3 (3L-384H) | 17.4M | 0.34 | 0.51 | 0.68 | **1.02** | 1.36 | ~✓ ≤ 90 |
| TinyBERT 4L-312D | 14.4M | 0.29 | 0.44 | 0.58 | **0.87** | 1.17 | ✓ @ ≤96 |
| Student 2L-256H | 9.6M | 0.10 | 0.15 | 0.20 | 0.30 | **0.40** | ✓ @ all |
| Student 4L-256H | 11.2M | 0.20 | 0.30 | 0.40 | **0.60** | **0.81** | ✓ @ all |

Notes the plan honors explicitly:
- **Params ≠ FLOPs**, demonstrated: ALBERT 11.7M params = **21.8G**; MobileBERT 19.5M params = only 3.9G. Always gate on the *measured* FLOPs, never on param count.
- **INT8 is NOT counted** as FLOP reduction. Treat quantization/ONNX strictly as a latency + model-size optimization, applied after the architecture+`max_length` decision to protect the 1 GFLOP budget.
- The FLOP/`max_length` trade-offs (32/48/64/96 view) are the columns above. Translation into truncation coverage: seq 32 → 67.8% of rows truncated; seq 48 → 25.4%; seq 64 → 8.6%; **seq 96 → 0.32%** (p99=85). Operating point recommendation: **96**, tiny fidelity cost, still inside budget for every viable model.

Per-candidate verdicts (compute / HF+FastAPI compat / effort):

1. **TinyBERT 4L-312D** (`huawei-noah/TinyBERT_4L_312D`) — Compute: ✓ 0.87G @96, 0.58G @64. Compat: HF hub checkpoint, loads via `AutoModelForSequenceClassification`, FastAPI placeholder identical to today. Effort: low (normal fine-tune), though it shares the BERT-uncased vocab so tokenizer carries over. High upside with the published TinyBERT distillation recipe.
2. **MiniLM-L3** (`microsoft/MiniLM-L3-H384-uncased`) — Compute: ✓ 0.68G @64; marginal 1.02G @96 (2% over → keep @90 to be safe, or accept 64). Compat: HF-native. Effort: very low. Good speed; moderate capacity headroom.
3. **MobileBERT** (`google/mobilebert-uncased`) — Compute: ✗ 3.94G @128, 1.97G @64, only 0.99G @32 — but @32 truncates 68% of your data → quality collapse. **Rejected on budget;** included just to document the measurement.
4. **ALBERT** (`albert-base-v2`) — Compute: ✗ never below ~5.4G. Shared weights shrink params but not FLOPs. **Rejected.**
5. **TF-IDF + Logistic Regression (baseline)** — Compute: ✓ trivially (a few MFLOPS). Compat: not HF; you'd keep a `TfidfVectorizer + Pipeline` object (joblib), which is FastAPI-compatible but *not* the HF pipeline path. Effort: already exists (acc 75.7%, macro-F1 67.9% — with `n_jobs` warning to fix). Quality per GFLOP is good but macro-F1 ceiling ~0.68 is far below the current 0.80.
6. **Custom 2–4-layer student distilled from your fine-tuned DistilBERT teacher** — Compute: ✓ 0.40–0.81G even @128 (4L-256H gives 0.81G @128 or 0.60G @96). Compat: 100% HF (`BertForSequenceClassification`), same FastAPI code path. Effort: moderate — needs teacher-logit caching + a KD training loop (logit-KL + hard-label CE + optional hidden-state alignment). This is the only path that keeps *both* a real margin under the budget *and* a known-good teacher/soft-target quality.

---

## D. Recommended Experiment Matrix (all under ≤1 GFLOP, macro F1 mandatory)

Fixed across all rows: single stratified 80/10/10 split (written once as `train/val/test` files, reused), 2 seeds, weighted CE default, evaluated via accuracy + macro F1 + per-class F1 + confusion matrix + the measurement block (GFLOP via `FlopCounterMode`, CPU latency p50/p95 over 1k forwards, model disk size, peak RAM).

| # | Model | max_len → est. GFLOP | Training | Purpose |
|---|---|---|---|---|
| E0 | DistilBERT (Drive FT) | 128 → 10.87G | (reuse) | Ceiling reference, not deployable |
| E1 | TF-IDF + LogReg | — (~0) | class_weight=balanced | Floor reference (0.68 macro F1) |
| R1 | **Student 4L-256H** | **96 → 0.60** | **KD from DistilBERT teacher + weighted CE** | **PRIMARY candidate** |
| R2 | Student 4L-256H | 128 → 0.81 | KD + weighted CE | Variant: no-truncation fidelity |
| R3 | Student 4L-256H | 96 → 0.60 | FT only (no KD) | KD ablative (“is the teacher worth it?”) |
| R4 | TinyBERT 4L-312D | 96 → 0.87 | KD + weighted CE | Higher-capacity fallback |
| R5 | MiniLM-L3 | 64 → 0.68 | FT only | Compromise: weakest macro-F1 likely |
| R6 | Student 2L-256H | 128 → 0.40 | KD + weighted CE | Ultra-light (if CPU latency dominates) |
| R7 | MobileBERT | 32 → 0.99 | FT | Probe only — document + reject |
| R8 | ALBERT-base | — (>5G) | — | Probe only — document + reject |

Within R1: micro-grid the loss weighting — (a) inverse-frequency weights `w̄ = N/(K·N_c)`, (b) sqrt scheme, (c) focal-γ=1. Pick by macro F1. Distillation settings: soft-target KL with T ∈ {3,5}, logits only (cheap), optional stage-2 hidden-state alignment on the 4L head only.

**Gating rule enforced at deploy:** measured forward FLOP ≤ 1e9 (1 GFLOP) on the *final exported* model; optionally re-verified on exported ONNX INT8 model for *latency/size* only (never to claim budget compliance).

---

## E. Recommended Path, Fallback, Files, Risks & Decisions

**Recommended first path:**
1. Taxonomy v2 remap → `news_domain_tagging_v2.csv` + `review_ambiguous.csv` (no original overwritten).
2. Freeze splits; run token-stat and FLOP-measure script non-negotiable at every milestone.
3. **Primary = Student 4L-256H (11.2M params), max_length=96 → 0.60 GFLOP**, distilled from the fine-tuned DistilBERT teacher (cached logits) + weighted CE, gate on macro F1 ≈ ≥0.80 within ±0.02 and per-class F1 report (watch Education/Crime).
4. Serve via FastAPI with the same `AutoModelForSequenceClassification` path used today; report GFLOP/latency/size/RAM; optional INT8/ONNX as a later latency/size pass.

**Fallback #1 (budget):** if the final measurement ever lands >1 GFLOP → step max_length 128→96→64 (Student 0.81 → 0.60 → 0.40; TinyBERT 0.87 → 0.58) before changing architecture.
**Fallback #2 (quality):** if macro F1 falls more than ~0.03 below the teacher → promote to R4 TinyBERT 4L-312D @96 (0.87G) with the same KD; if still short → raise KD to hidden-state alignment or add the review-bucket data back; R6 (2L-256H) is the last-resort only if CPU latency is unacceptable.

**Files that will change / need to exist (flag: frontend/backend not found in workspace):**
- `01_data_loading.ipynb`, `Copy of Untitled0.ipynb` (parent folder) — keep as reference; New colored workflows go in parallel scripts (notebook edits kept minimal).
- New: `data/taxonomy.py` (label rules + mapping table), `data/remap_dataset.py` (writes v2 + review CSVs + mapping report), `analysis/token_stats.py`, `analysis/measure_flops.py` (the patched `FlopCounterMode` script), `train/cache_teacher_logits.py`, `train/train_student.py`, `eval/evaluate.py`, `deploy/model_export.py` (+ optional `deploy/onnx_quantize.py`).
- Missing & must be located/created: **FastAPI `app.py` (HF model loader), React frontend, and the deployed model dir/checkpoint sync from Google Drive.**

**Risks, assumptions, and decisions needed before coding:**
1. **Budget wording:** assume **≤1 GFLOP** per forward (FLOP = 2×MACs, torch convention) per user note. If instead meant MACs (≈0.5 GFLOP) or a soft “≥1” bound, the viable cells shift (R4's 0.87 and R2's 0.81 become problematic). Confirm.
2. **Teacher access:** the fine-tuned DistilBERT checkpoint only exists in Google Drive. Distillation needs it. Choose: (a) sync it locally, (b) use stock `distilbert-base-uncased` as teacher, or (c) allow teacher re-training later (contradicts “don't retrain yet” — needs approval).
3. **Compute budget for training:** this machine is CPU-only torch (`2.12.1+cpu`). Students (11–14M params) can train overnight on CPU (~1–4 h), heavier runs far better on the Colab T4 already used. Confirm where training runs.
4. **Taxonomy decisions:** keep exactly 8 labels? Replace “World” with **International News** (geography-of-event rule) as proposed? Add `Religion` / `Science` slots instead of folding into Lifestyle/Politics? Confirm ambiguous-bucket handling (manual review vs keyword-split vs drop) — especially identity-VOICES sections previously force-mapped to Politics.
5. **Resurrect the 20,732 dropped rows** under the new taxonomy (recommended yes — helps the 2 worst classes).
6. **max_length:** recommend 96 (0.32% truncated) as operating point for the primary student; confirm 128 is not required.
7. **Frontend/backend/checkpoint location:** the React + FastAPI + local-model pieces are not in this workspace — point to them (or confirm they should be built fresh).
8. **Latency target & hardware:** define the CPU spec for the latency benchmark so R6/R1 selection is meaningful.
9. **Loss regime:** confirm default = inverse-frequency class weights + optional focal; 2-seed reporting acceptable.
10. **INT8/ONNX:** out-of-scope for the FLOP gate (per instruction) but confirm may be added later for latency/size.

---

## Appendix: Exact measurements made on 2026-08-17 (this machine, venv)

- Env: `transformers 5.12.1`, `torch 2.12.1+cpu`, `sklearn 1.9.0`, `datasets 5.0.0`, Python 3.13.5 (`venv`).
- `distilbert-base-uncased` present in HF cache (`C:\Users\Asus\.cache\huggingface`); tokenizer + model used offline.
- Token-length stats: computed on all 188,795 texts with the DistilBERT tokenizer (`truncation=False`, add_special_tokens=True).
- FLOP measurement: `FlopCounterMode` per the patched method above; per-length values are consecutive cumulative differences.
- Mapped rows path note: raw `category` column is intact in the CSV for taxonomy v2 re-derivation.