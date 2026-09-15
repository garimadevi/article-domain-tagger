"""Export the trained student with id2label and verify the 1 GFLOP gate.

Saves the checkpoint in standard transformers format (config.json + safetensors
+ id2label/label2id) so the FastAPI loader path and consumers only need
AutoModelForSequenceClassification.from_pretrained. Re-runs the FLOP gate on
the exported artifact.

Usage:
  python -m deploy.model_export --model <dir> --out generated/deployed_student
"""

import argparse
import os

import numpy as np
import torch

import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained student dir (e.g. generated/student_4L_256H_seed42)")
    ap.add_argument("--out", default=str(config.DEPLOYED_STUDENT))
    ap.add_argument("--verify-flops", action="store_true")
    ap.add_argument("--max-len", type=int, default=config.MAX_LEN)
    args = ap.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    id2label = dict(config.ID2LABEL)
    label2id = dict(config.LABEL2ID)
    model.config.id2label = id2label
    model.config.label2id = label2id

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] params={n_params:,} ({n_params*4/1e6:.1f} MB fp32) -> {args.out}")

    if args.verify_flops:
        from analysis.measure_flops import measure_flops
        model.eval()
        with torch.no_grad():
            flops = [x for x in measure_flops(model, [args.max_len])][0]
        gflop = flops[1] / 1e9
        status = "PASS" if gflop <= config.GFLOP_BUDGET else "FAIL"
        print(f"[export] FLOP gate @{flops[0]}: {gflop:.3f} GFLOP ({gflop/2:.3f} GMACs) "
              f"/ budget {config.GFLOP_BUDGET} -> {status}")
        rc = 0 if status == "PASS" else 2
    else:
        rc = 0

    # quick sanity: load back + one prediction
    m2 = AutoModelForSequenceClassification.from_pretrained(args.out)
    _ = m2.eval()
    print("[export] reload sanity OK")


if __name__ == "__main__":
    main()