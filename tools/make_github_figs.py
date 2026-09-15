"""Generate GitHub Pages figures from existing project data (no retraining).

Reads:
  generated/token_stats.csv
  generated/class_counts_revised.csv
Uses hardcoded REPORT.md test metrics for per-class F1 + teacher-vs-student.

Writes PNGs to docs/assets/images/:
  token_dist.png, per_class_f1.png, teacher_vs_student.png,
  class_imbalance.png, architecture.png

Usage:
  venv\\Scripts\\python.exe tools\\make_github_figs.py
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def save(fig, name):
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figs] wrote -> {p}")
    return p


def fig_token_dist():
    csv = ROOT / "generated" / "token_stats.csv"
    df = pd.read_csv(csv)
    tr = df[df["stat"] == "truncation"].copy()
    tr["max_len"] = tr["max_len"].astype(int)
    tr = tr.sort_values("max_len")
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(tr["max_len"].astype(str), tr["truncated_frac"] * 100, color="#2563eb")
    ax.set_xlabel("max_len (tokens)")
    ax.set_ylabel("truncated % of articles")
    ax.set_title("Truncation vs max_len (DistilBERT tokenizer, n=167K)")
    for b, v in zip(bars, tr["truncated_frac"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{v:.2f}%", ha="center", fontsize=8)
    ax.annotate("Chosen: 96 (0.28% trunc, p99=85)", xy=(3, tr["truncated_frac"].iloc[3] * 100),
                xytext=(1.5, 30), arrowprops=dict(arrowstyle="->"), fontsize=9, color="#1d4ed8")
    return save(fig, "token_dist.png")


def fig_per_class_f1():
    labels = ["Sports", "Health", "Politics & Govt", "Crime & Justice", "Economy & Biz",
              "Disasters", "War & Conflicts", "Sci & Tech", "Environment", "Education", "Law & Legal"]
    f1 = [0.958, 0.870, 0.866, 0.837, 0.818, 0.786, 0.772, 0.723, 0.663, 0.601, 0.537]
    colors = ["#22c55e" if v >= 0.8 else "#3b82f6" if v >= 0.7 else "#f59e0b" if v >= 0.6 else "#ef4444" for v in f1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = range(len(labels))
    bars = ax.barh(list(y), f1, color=colors)
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("Test F1")
    ax.set_xlim(0, 1.0)
    ax.set_title("Per-class F1 (test, 6,509 articles) — Sports 0.958 to Law 0.537")
    for b, v in zip(bars, f1):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    ax.invert_yaxis()
    return save(fig, "per_class_f1.png")


def fig_teacher_vs_student():
    cats = ["Params (M)", "GFLOP @96", "Macro-F1"]
    teacher = [67.0, 8.16, 0.807]
    student = [11.2, 0.64, 0.766]
    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar([i - w / 2 for i in x], teacher, width=w, label="Teacher DistilBERT", color="#8b5cf6")
    b2 = ax.bar([i + w / 2 for i in x], student, width=w, label="Student 4L-256H", color="#2563eb")
    ax.set_xticks(list(x), cats)
    ax.set_title("Teacher vs Student — 95% F1 at 1/12 compute")
    ax.legend()
    for b, v in list(zip(b1, teacher)) + list(zip(b2, student)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02, f"{v}", ha="center", fontsize=8)
    return save(fig, "teacher_vs_student.png")


def fig_class_imbalance():
    csv = ROOT / "generated" / "class_counts_revised.csv"
    df = pd.read_csv(csv)
    # shorten long labels for display
    short = {"Lifestyle and Leisure": "Lifestyle", "Politics and Government": "Politics",
             "Arts, Culture, Entertainment and Media": "Arts/Entertain.",
             "Economy, Business and Finance": "Economy/Biz", "Sport": "Sports",
             "Crime, Law and Justice": "Crime/Law"}
    df["disp"] = df["revised_category"].map(short).fillna(df["revised_category"])
    df = df.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh(df["disp"], df["count"], color="#0ea5e9")
    ax.set_xlabel("Articles")
    ax.set_title("Class imbalance — why inverse-frequency weights (w_c) are mandatory")
    for b, v in zip(bars, df["count"]):
        ax.text(v * 1.01, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=8)
    return save(fig, "class_imbalance.png")


def fig_architecture():
    """Flowchart-style architecture rendered with matplotlib (no mermaid dependency)."""
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    boxes = [
        (0.2, "Raw IAB\n106K articles", "#dbeafe"),
        (2.3, "Filter 11 labels\n+ Env keywords\n65K articles", "#e0e7ff"),
        (4.4, "Split 80/10/10\n52K/6.5K/6.5K\nseed 42, len 96", "#fef3c7"),
        (6.5, "Teacher DistilBERT\n67M, 6L-768H\nval F1 0.807", "#ede9fe"),
        (8.0, "Cache logits\nsoft labels\nT=3", "#fce7f3"),
        (9.2, "Student 4L-256H\n11.2M, KD a=0.7\nval F1 0.771", "#dcfce7"),
        (10.7, "44MB, 0.64G\nFastAPI+React", "#cffafe"),
    ]
    # draw boxes + arrows
    for i, (x, txt, col) in enumerate(boxes):
        w = 1.15 if i not in (1, 2) else 1.35
        if i == 1:
            x = 1.7
        if i == 2:
            x = 3.35
        if i >= 3:
            x = 5.0 + (i - 3) * 1.42
            w = 1.3
        # Rectangle has no boxstyle; use FancyBboxPatch instead
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        box = FancyBboxPatch((x, 1.2), w, 1.6, boxstyle="round,pad=0.05", facecolor=col,
                             edgecolor="#334155", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w / 2, 2.0, txt, ha="center", va="center", fontsize=7.5, weight="bold")
        if i < len(boxes) - 1:
            # arrow to next box start
            nxt_x = [1.7, 3.35, 5.0, 6.42, 7.84, 9.26, 10.68][i + 1]
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 2.0), (nxt_x - 0.02, 2.0),
                                         arrowstyle="->", mutation_scale=12, linewidth=1.4, color="#334155"))
    ax.set_title("Article Domain Tagger — Teacher -> Distillation -> Student -> Deploy pipeline", fontsize=11, weight="bold", pad=12)
    fig.text(0.5, 0.06, "DistilBERT 67M (8.16 GFLOP FAIL)  ->  Student 4L-256H 11.2M (0.64 GFLOP PASS, 95% F1 retained)",
             ha="center", fontsize=9, color="#475569")
    return save(fig, "architecture.png")


def main():
    print(f"[figs] root={ROOT} out={OUT}")
    fig_token_dist()
    fig_per_class_f1()
    fig_teacher_vs_student()
    fig_class_imbalance()
    fig_architecture()
    print("[figs] done. Update docs/index.html image links if renamed.")


if __name__ == "__main__":
    main()
