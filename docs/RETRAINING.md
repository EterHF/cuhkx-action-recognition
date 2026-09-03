# Retraining strictV3

[English](RETRAINING.md) | [简体中文](RETRAINING.zh-CN.md) | [Documentation index](README.md) | [Repository home](../README.md)

## What counts as a retrain

`release.verify` and `release.replay` validate frozen weights. They are not
training runs. A retrain must create new checkpoints from the declared
bootstrap assets, regenerate held-user logits, refit release calibration using
outer-train rows only, and finally run raw inference with the newly packed
model.

strictV3 is a **release-strict** baseline. Its public R(2+1)D-34 and DSTFormer
bootstrap assets were previously fitted with CUHK-X labels. Consequently its
0.956192 OOF value is an engineering replay metric, not an unbiased estimate
for a model initialized only from external data. The distinction from the
fully-external protocol is retained in the technical report.

## Fixed folds

| Fold | Held users | Rows |
| --- | --- | ---: |
| A | 1, 6, 17, 22 | 716 |
| B | 2, 7, 18, 23 | 673 |
| C | 3, 8, 19, 24 | 679 |
| D | 4, 9, 20 | 489 |
| E | 5, 16, 21 | 479 |

The held users may be read for the final OOF measurement only. Temperature and
branch-weight fitting for a held fold use the other four folds.

## Input contract

The competition data are not redistributed. Build the visual and skeleton
caches from `data/processed/{train,test}` with the modules under
`yolo_r2plus1d.strict_v3.data`. Before training, record SHA-256 hashes for the
following inputs:

- `train_depth_ir.npy` and `test_depth_ir.npy`;
- `train_skeleton.npy`, `test_skeleton.npy`, and both validity masks;
- `train_frame_logits.npy` and `test_frame_logits.npy` from the declared
  DSTFormer bootstrap;
- `metadata.npz`, the public visual bootstrap package, and any fusion resume
  checkpoint.

The canonical temporal inputs have hashes:

```text
train_frame_logits.npy  76e6c11c86a681f03f21f7114eae17b1bc8328e6cd6a161f365b8b0cfc37102e
test_frame_logits.npy   b96677c5bc7adc2ef70f53c5819d1042dc8585df2ee4e67965031f10c8db4fbb
metadata.npz            bf2e93e558b4ae148b14835c1f65c943828bff97dfaaf224a39c249c7599f099
```

The suite runner embeds absolute input paths and hashes in `receipt.json`.
This makes a run auditable without treating a checkpoint filename as proof of
provenance.

## Temporal baseline

Run all five folds and the full epoch-5 model:

```bash
CUDA_VISIBLE_DEVICES=0 .conda/envs/cuhkx/bin/python \
  -m yolo_r2plus1d.strict_v3.training.temporal_suite \
  --recipe strict \
  --frame-logits .cache/strict_v3/train_frame_logits.npy \
  --test-frame-logits .cache/strict_v3/test_frame_logits.npy \
  --metadata results/strict_v3/metadata.npz \
  --output runs/strict_v3/temporal
```

The historical fold accuracies are A–E = 0.937151, 0.933135, 0.945508,
0.959100 and 0.926931; aggregate temporal OOF is 0.940053. Different devices
may produce non-byte-identical logits even when every fold accuracy matches.
Therefore compare the per-fold metrics and prediction differences as well as
the checkpoint hash.

## sched30 experiment

The experiment fixes the checkpoint at epoch 5 while retaining a 30-epoch
cosine schedule. It uses seeds 2026, 2027 and 2028; validation labels are read
once after the fixed checkpoint is written.

```bash
CUDA_VISIBLE_DEVICES=0 .conda/envs/cuhkx/bin/python \
  -m yolo_r2plus1d.strict_v3.training.temporal_suite \
  --recipe sched30 \
  --frame-logits .cache/strict_v3/train_frame_logits.npy \
  --test-frame-logits .cache/strict_v3/test_frame_logits.npy \
  --metadata results/strict_v3/metadata.npz \
  --output runs/experiments/sched30
```

`--device cpu --workers 0` is supported for an independent numerical check.
CPU and A100 runs can choose adjacent blend-grid points when their outer-train
scores tie, so deployment calibration must record both the selected weights
and the device. A candidate passes only if its aggregate, macro, subject-macro,
worst-user and worst-fold metrics do not regress, at least four folds do not
regress, and raw replay matches the newly generated submission.

## Observed retraining audit (2026-09-03)

The following jobs were actual training runs, not frozen-checkpoint replays.
They ran serially on one A100 while an unrelated service occupied most of the
card. `public_finetune` and `fusion` now accept
`--cuda-memory-fraction 0.14`; this was sufficient for visual head-only
batch 8 and fusion batch 8. Both entry points also accept `--deterministic`.
The historical recipe did not enable deterministic cuDNN, so equality is
judged by fold accuracy and prediction differences, not checkpoint hashes.

| Fold | Temporal new / historical | Visual new / historical | Fusion new / historical |
| --- | ---: | ---: | ---: |
| A | 0.937151 / 0.937151 | 0.960894 / 0.960894 | 0.959497 / 0.959497 |
| B | 0.933135 / 0.933135 | 0.888559 / 0.888559 | 0.974740 / 0.974740 |
| C | 0.945508 / 0.945508 | 0.979381 / 0.983800 | 0.718704 / 0.718704 |
| D | 0.959100 / 0.959100 | 0.860941 / 0.856851 | 0.862986 / 0.862986 |
| E | 0.926931 / 0.926931 | 0.776618 / 0.776618 | 0.874739 / 0.874739 |

Temporal aggregate OOF was 0.940053. Against the historical validation
predictions, visual argmax differences were A–E = 6, 0, 5, 9 and 4; fusion
differences were 0, 1, 4, 14 and 0. Keeping `--workers 8` is part of the
historical visual recipe because horizontal flips are sampled inside loader
workers.

The independent CPU `sched30` suite completed three seeds × five folds plus
three full models. Its temporal probability mean scored 0.937747 OOF. When
recalibrated with the frozen visual/fusion branches, the candidate scored
0.959486 (+0.003294 over strictV3, 5/5 folds non-degrading). This is below the
separately frozen 0.960474 candidate: CPU/GPU numerical differences moved one
outer-train grid tie to an adjacent weight. The retrained result is therefore
evidence of potential, not a promoted release.

The audit is still **partial**: a new full fusion deployment package, two
byte-identical raw replays and an authorized Kaggle confirmation have not been
completed. The canonical 0.97512 package remains unchanged.

## Final acceptance

A completed strictV3 retrain requires all of the following:

1. New visual, fusion and temporal checkpoints for folds A–E and the full
   deployment path.
2. Fold-aligned OOF arrays with exactly 3,036 rows and no missing row.
3. Calibration fitted without each fold's labels and a natural 40-class test
   prediction set.
4. Model plus YOLO size at most 100,000,000 bytes.
5. Two raw replays producing identical CSV hashes.
6. A Kaggle submission only after the local gate passes; the returned public
   score is recorded as external evidence, never used to retune the model.

Until every item passes, the repository must describe the run as partial and
must not replace the canonical 0.97512 release.
