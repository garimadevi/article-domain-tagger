# Video Script — Article Domain Tagger (6–7 min, 1080p)

> Record: OBS Studio or Win+G. Mic + system audio. 1920×1080, show code at 125% zoom.
> Upload: YouTube Studio → Title below → Description below → Unlisted/Public → copy 11-char ID → replace `YOUR_VIDEO_ID` in `docs/index.html` and `README.md`.

**Title:** `Article Domain Tagger — Knowledge Distillation (11 Domains, 0.64 GFLOP) | Garima Devi`

**Description (paste):**
```text
Article Domain Tagger: classify news into 11 IAB domains with a 44MB distilled student (4L-256H, 0.64 GFLOP, 82.4% accuracy, 0.766 macro-F1).

By Garima Devi.
Mentors: Assigned Mentor, Mohd. Amaan Sir.
Faculty Advisor: Prof. Prithwijit Guha. Co-Advisor: Ashwin Jacob Gigo.

GitHub: https://github.com/garimadevi/article-domain-tagger
Project page: https://garimadevi.github.io/article-domain-tagger/
Dataset: https://huggingface.co/datasets/mdonigian/iab-news-classification
Teacher: distilbert-base-uncased (67M) -> Student 4L-256H (11.2M), KD T=3 alpha=0.7.

Chapters in pinned comment.
```

## Chapters & script

**0:00 Intro (30s)** — Show `docs/index.html` hero. Say: title, your name (Garima Devi), mentors (Assigned Mentor, Mohd. Amaan Sir), advisors (Prof. Prithwijit Guha, Ashwin Jacob Gigo), deadline 15 Sept 2026, metrics badges.

**0:30 Problem & dataset (1 min)** — Open `data/taxonomy.py`, `REPORT.md` §2–3. Why 11 IAB labels (no vague "World"), Environment keyword extraction, 106K→65K, split 80/10/10 seed 42.

**1:30 Architecture (1.5 min)** — Show `docs/assets/images/architecture.png` + `tools/architecture.mmd`. Walk 7 boxes. Stress: teacher 8.16 GFLOP FAIL vs student 0.64 PASS.

**2:30 Logic / code (1.5 min)** — Open `train/train_student.py:38-43` (`kd_loss`), `train/common.py:40-46` (`w_c`), `train/student.py` (BertConfig 4L-256H), `config.py:51` (budget). Show equations on page rendering via MathJax.

**4:00 Results & graphs (1 min)** — Show `REPORT.md` results table + 4 PNGs in `docs/`. Teacher 0.807 → student 0.771/0.766, per-class Sports 0.958 → Law 0.537 (small classes), token p99=85 justifies len 96.

**5:00 Live demo (1 min)** — Terminal 1: `venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000`. Terminal 2: `cd frontend; npm run dev`. Paste 2 texts (politics + sports examples from `frontend/src/App.jsx:106-116`), show top-3 + confidence, optionally `POST /fetch-url` with a news URL.

**6:00 Limitations & close (30s)** — Law/Edu weak, 42K ambiguous rows, keyword-Env noise, future (manual labeling, T=5, ONNX, threshold). Show footer links, ask to star repo.

## Recording checklist

- [ ] Close notifications, 125% zoom, dark code theme readable
- [ ] Test mic 10s, record system audio for demo clicks
- [ ] One clean take; minor ums OK — deadline over perfection
- [ ] Export 1080p mp4, upload, set chapters in description from times above
