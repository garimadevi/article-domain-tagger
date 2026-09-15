# Article Domain Tagging — Training Progress

> Continuation of `SESSION_LOG_2026-08-18.md` (data prep, analysis, pipeline build, and smoke tests).
> This file records today's session: retraining the teacher from 8 labels to 10 labels.

## Objective

Expand the existing DistilBERT teacher model from 8 article-domain labels to 10 labels and retrain it.

## Environment

- Google Colab
- GPU: NVIDIA T4 / CUDA
- Project path:
  `/content/drive/MyDrive/ArticleTagging_v2/project_code`
- Original teacher checkpoint:
  `/content/drive/MyDrive/ArticleTagging_v2/old_teacher_8_labels`
- New saved teacher checkpoint:
  `/content/drive/MyDrive/ArticleTagging_v2/outputs/distilbert_teacher_revised_10_labels`

## Problem Encountered

The original 8-label classifier head could not be loaded into a 10-label model. The error was:

```
RuntimeError: You set `ignore_mismatched_sizes` to `False`
```

Transformers reported classifier shape mismatches:

- Old classifier weight: `torch.Size([8, 768])`
- New classifier weight: `torch.Size([10, 768])`
- Old classifier bias: `torch.Size([8])`
- New classifier bias: `torch.Size([10])`

## Code Change Applied

Updated:

```
train/teacher_train.py
```

Changed the teacher model loading call from:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    args.teacher_from, num_labels=len(config.LABELS)
)
```

to:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    args.teacher_from,
    num_labels=len(config.LABELS),
    ignore_mismatched_sizes=True
)
```

Reason: retain the pretrained DistilBERT encoder while reinitializing only the final classifier head from 8 outputs to 10 outputs.

## Training Command

```
PYTHONPATH=. python -m train.teacher_train \
  --teacher-from "/content/drive/MyDrive/ArticleTagging_v2/old_teacher_8_labels" \
  --out "/content/drive/MyDrive/ArticleTagging_v2/outputs/distilbert_teacher_revised_10_labels" \
  --device cuda \
  --max-len 96 \
  --batch-size 16 \
  --epochs 3 \
  --lr 2e-5 \
  --seed 42 \
  --weights-scheme inverse \
  --save-every-epoch
```

## Dataset

- Training examples: 133,615
- Validation examples: 16,701
- Number of labels: 10
- Class weighting scheme: inverse-frequency class weights

## Results

| Epoch | Validation Loss | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| 1 | 0.6167 | 0.8255 | 0.7440 | 0.8347 |
| 2 | 0.3987 | 0.8496 | 0.7763 | 0.8529 |
| 3 | 0.2712 | 0.8407 | 0.7665 | 0.8446 |

## Best Model

- Best epoch: 2
- Best validation macro-F1: 0.7763
- Best validation accuracy: 0.8496
- Best validation weighted-F1: 0.8529
- The script saved the best checkpoint automatically at:
  `/content/drive/MyDrive/ArticleTagging_v2/outputs/distilbert_teacher_revised_10_labels`

Note that epoch 3 had lower training loss but worse validation macro-F1, accuracy, and weighted-F1 than epoch 2, indicating mild overfitting. The saved checkpoint should therefore correspond to epoch 2, not epoch 3.

## Status

- [x] Review prior 8-label teacher checkpoint
- [x] Expand label configuration to 10 classes
- [x] Fix checkpoint-loading mismatch for 8-label to 10-label transition
- [x] Train the revised 10-label teacher model
- [x] Save the best validation checkpoint
- [x] Confirm best epoch based on validation macro-F1
- [ ] Inspect saved model files and verify config has num_labels = 10
- [ ] Evaluate the best model on the held-out test split
- [ ] Generate teacher logits/probabilities if knowledge distillation is the next stage
- [ ] Train and evaluate the student model
- [ ] Prepare final comparison/report