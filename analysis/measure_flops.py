"""FLOP / latency / size measurement for the 1 GFLOP budget gate.

Patches the installed FlopCounterMode table (missing tabulate in this venv),
measures forward FLOPs (torch convention = 2xMACs) at a set of candidate
sequence lengths, plus CPU latency p50/p95 over 1k forwards, param count,
model size on disk, and peak RAM.

Usage:
  python -m analysis.measure_flops --arch student [--layers 4 --hidden 256]
      --arch student     : build the Student 4L-256H config
      --arch teacher     : load distilbert-base-uncased (stock) with 10 labels
      --path <dir>       : load a saved HF checkpoint (teacher / student)
"""

import argparse
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

FlopCounterMode.get_table = lambda *a, **k: ""  # missing tabulate patch

import config


def build_model(arch: str, path: str | None, num_labels: int, layers: int, hidden: int):
    if path:
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(path)
    if arch == "teacher":
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    if arch == "student":
        from train.student import build_student
        return build_student(num_labels=num_labels)
    if arch == "mini":
        from train.student import student_config
        from transformers import BertForSequenceClassification
        return BertForSequenceClassification(student_config(num_labels=num_labels, layers=layers, hidden=hidden))
    raise ValueError(f"unknown arch: {arch}")


def measure_flops(model, lengths, device="cpu"):
    """Per the plan's verified method: single context, read cumulative
    get_total_flops() after each length, take consecutive differences."""
    model.eval()
    out = []
    with FlopCounterMode(display=False) as fm:
        prev = 0
        for L in lengths:
            ids = torch.ones(1, L, dtype=torch.long, device=device)
            am = torch.ones_like(ids)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=am)
                total = fm.get_total_flops()
            out.append((L, total - prev))
            prev = total
    return out


def measure_latency(model, L, num_iters=1000, device="cpu"):
    model.eval()
    ids = torch.ones(1, L, dtype=torch.long, device=device)
    am = torch.ones_like(ids)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(input_ids=ids, attention_mask=am)
        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            model(input_ids=ids, attention_mask=am)
            times.append(time.perf_counter() - t0)
    times = sorted(times)
    n = len(times)
    return times[n // 2], times[int(n * 0.95)], sum(times) / n


def peak_ram_mb():
    import tracemalloc
    tracemalloc.start()
    try:
        tracer = torch.zeros(1024, 1024)
        data = [tracer.sum().item() for _ in range(1024)]
    finally:
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak / 1e6


def model_size_mb(model, path=None):
    if path:
        import pathlib
        size = 0
        for f in pathlib.Path(path).rglob("*.safetensors"):
            size += f.stat().st_size
        return size / 1e6
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["student", "mini", "teacher", "path"], default="student")
    ap.add_argument("--path")
    ap.add_argument("--num-labels", type=int, default=10)
    ap.add_argument("--layers", type=int, default=config.NUM_LAYERS)
    ap.add_argument("--hidden", type=int, default=config.HIDDEN_SIZE)
    ap.add_argument("--lengths", default="32,48,64,96,128")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    model = build_model(args.arch, args.path, args.num_labels, args.layers, args.hidden)
    model = model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch} path={args.path} params={n_params:,} device={args.device}")

    print("\nforward FLOP measurements (torch convention  2xMACs):")
    print("  len    GFLOP     GMACs")
    for L, flops in measure_flops(model, lengths, device=args.device):
        print(f"  {L:3d}  {flops/1e9:8.3f}  {flops/2e9:8.3f}   (2xMACs; budget gate: GFLOP<=1.0)")

    L = max(min(lengths), 96)
    p50, p95, mean = measure_latency(model, L, args.iters, device=args.device)
    print(f"\nlatency @ {L} (CPU bench relevance; device={args.device}):")
    print(f"  p50={p50*1e3:.1f} ms  p95={p95*1e3:.1f} ms  mean={mean*1e3:.1f} ms  ({args.iters} fwd)")
    print(f"  params={n_params:,}  peak_traced_RAM~={peak_ram_mb()/1024:.1f} GB (rough)")
    print(f"  fp32 weights ~ {n_params*4/1e6:.1f} MB")


if __name__ == "__main__":
    main()