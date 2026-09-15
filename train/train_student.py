"""Train the Student 4L-256H via knowledge distillation from the teacher.

Primary candidate R1: Student 4L-256H, max_len=96 -> 0.604 GFLOP.
Loss = alpha * KL(soft targets, logits/T) + (1-alpha) * CE(hard labels) weighted.
Teacher soft targets come from cached NPZ (train/cache_teacher_logits.py).
2 seeds reported; best checkpoint saved by val macro-F1.

Designed for Colab T4:
  python -m train.train_student \
      --logits "<dir>/train_logits.npz" --out "generated/student_4L_256H"

Smoke test (local CPU):
  python -m train.train_student --logits generated/teacher_logits_smoke/train_logits.npz \
      --sample 200 --epochs 1 --batch-size 8 --out generated/student_smoke
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

import config
from train.common import (
    add_common_args, device_from_args, class_weights, load_split, make_loaders,
)
from train.student import build_student


def load_logits(npy_path, fp16_npz=False):
    data = np.load(npy_path)
    logits = data["logits"]
    return torch.tensor(logits, dtype=(torch.float16 if fp16_npz else torch.float32))


def kd_loss(student_logits, teacher_logits, labels, weights, T=3.0, alpha=0.7, device="cpu"):
    ce = torch.nn.functional.cross_entropy(student_logits, labels, weight=weights.to(device))
    s = torch.log_softmax(student_logits / T, dim=-1)
    t = torch.softmax(teacher_logits / T, dim=-1)
    kl = (t * (torch.log(t + 1e-9) - s)).sum(-1).mean()
    return alpha * (T * T) * kl + (1.0 - alpha) * ce, ce.item(), kl.item()


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        preds.append(out.logits.argmax(-1).cpu().numpy())
        labels.append(batch["label"].numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def run_training(seed, args, device, tokenizer, train_df, val_df, train_loader, val_loader, teacher_logits, weights, out_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_student(num_labels=len(config.LABELS), seed=seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_ce, total_kl, n_batches = 0.0, 0.0, 0.0, 0
        step = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            start = step * args.batch_size
            t_logits = teacher_logits[start:start + len(label)].to(device)
            step += 1
            opt.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn)
            loss, ce, kl = kd_loss(out.logits, t_logits, label, weights, args.temp, args.alpha, device)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_ce += ce
            total_kl += kl
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        m = eval_model(model, val_loader, device)
        print(f"[seed {seed}] epoch {epoch}/{args.epochs} loss={avg_loss:.4f} (ce={total_ce/n_batches:.4f} "
              f"kl={total_kl/n_batches:.4f}) acc={m['acc']:.4f} f1_macro={m['f1_macro']:.4f}")
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_f1, best_state


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--logits", required=True, help="path to train_logits.npz from cache step")
    ap.add_argument("--out", default=str(config.STUDENT_OUT))
    ap.add_argument("--teacher-logits-dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--temp", type=float, default=3.0, help="soft-target temperature")
    ap.add_argument("--alpha", type=float, default=0.7, help="KL weight (1-alpha = CE weight)")
    ap.add_argument("--seeds", default=config.SEEDS_ARG, help="comma-separated seeds; each reported")
    ap.add_argument("--load-best-after", type=int, default=None,
                    help="reload last saved best for a further report")
    args = ap.parse_args()

    device = device_from_args(args)
    print(f"[student] device={device} config: layers=4 hidden=256 max_len={args.max_len} "
          f"temp={args.temp} alpha={args.alpha} seeds={args.seeds}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_df = load_split(args.data_dir, "train", args.sample)
    val_df = load_split(args.data_dir, "val", args.sample)
    weights = class_weights(train_df["label"], scheme="inverse")
    print(f"[student] train={len(train_df)} val={len(val_df)}")
    print(f"[student] class weights: {weights.tolist()}")

    teacher_logits = load_logits(args.logits, fp16_npz=(args.teacher_logits_dtype == "fp16"))
    assert teacher_logits.shape[0] == len(train_df), \
        f"logits rows {teacher_logits.shape[0]} != train rows {len(train_df)}"
    print(f"[student] teacher logits {teacher_logits.shape} dtype {teacher_logits.dtype}")

    # shared loaders use `steps` aligned to batch; we require shuffle sampler staying
    # in split order so teacher_logits[i*bs:(i+1)*bs] aligns — build ordered loader.
    train_loader = make_loaders(train_df, tokenizer, args.max_len, args.batch_size, args.workers,
                                shuffle=False, device=str(device))
    val_loader = make_loaders(val_df, tokenizer, args.max_len, args.batch_size, args.workers,
                              shuffle=False, device=str(device))

    seeds = [int(s) for s in args.seeds.split(",")]
    results = {}
    for seed in seeds:
        out_dir = f"{args.out}_seed{seed}"
        os.makedirs(out_dir, exist_ok=True)
        best_f1, best_state = run_training(seed, args, device, tokenizer,
                                           train_df, val_df, train_loader, val_loader,
                                           teacher_logits, weights, out_dir)
        model = build_student(num_labels=len(config.LABELS), seed=seed)
        model.load_state_dict(best_state)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        results[seed] = best_f1
        print(f"[student] seed {seed}: best val f1_macro={best_f1:.4f} saved -> {out_dir}")

    if len(seeds) > 1:
        vals = list(results.values())
        print(f"[student] final: seeds={results} mean best f1_macro={np.mean(vals):.4f} "
              f"sd={np.std(vals):.4f}")


if __name__ == "__main__":
    main()