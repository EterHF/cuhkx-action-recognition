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

`data/external/` is explicitly outside this strictV3 retraining contract.
NTU RGB+D and PKU-MMD are not needed to reproduce the canonical 0.97512
package. A fully-external research run must use a separate manifest, initialise
a fresh 40-way head inside every outer fold, and keep its outputs outside
`checkpoints/strict_v3/`. External archives and source-derived experimental
weights are not redistributed by this repository. This separation is a
provenance and release rule, not a prohibition on training: the competition
host [allows publicly obtainable external data and pretrained models and
explicitly permits NTU RGB+D](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/724404).
Request-form access is also allowed when open to anyone. Record the provider,
access steps, manifest/hash, preprocessing, and role of every external source
in the final writeup.

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

For a confirmatory OOF screen that must not touch anonymous test assets, omit
`--test-frame-logits` and add `--oof-only`. This mode refuses a test-logit
argument, skips all full-data jobs, writes `null` for test paths and hashes,
and records `test_data_loaded: false` in `receipt.json`.

## Observed retraining audit (through 2026-09-04)

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

Three further temporal generalisation hypotheses were preregistered and run as
train-only screens (3 seeds × 5 folds, CPU, fixed epoch 5, no test input):

| Candidate | Temporal seed-mean | Fold / seed stability | Decision |
| --- | ---: | --- | --- |
| Epoch-resampled reversal/noise | 0.937747 (2,847/3,036; no change) | 5/5 folds and 3/3 seeds non-degrading | Rejected at stage 1: required at least 2,848 correct rows |
| Uniform FP32 weight mean, epochs 3–5 | 0.938076 (2,848/3,036; +1 row) | 4/5 folds and 3/3 seeds non-degrading | Passed temporal gate; rejected at release gate |
| Same-class cross-user temporal-residual mix, fixed 0.25 | 0.937747 (2,847/3,036; no top-1 change) | 5/5 folds and 3/3 seeds tied | Rejected at stage 1: required at least 2,848 correct rows |

For the weight-average candidate, macro recall improved by 0.000327 and
subject-macro by 0.000208; worst-user and worst-fold were unchanged. Its
train-only nested branch reached 0.961792 (2,920 rows), one row above the
historical nested sched30 branch. The preregistered fixed 50/50 probability
ensemble with strictV3 nevertheless remained prediction-identical to the
existing candidate: 0.960474 (2,916 rows), zero changed predictions,
against a required 2,917 rows. Historical control replay reproduced both the
nested logits and final probability array exactly. The stop rule therefore
prevented full-data training, test inference and packaging.

The cross-user mix selected a same-class partner from a different outer-train
subject and mixed 25% of its zero-mean temporal residual into the anchor clip.
All checkpoints and OOF logits changed, but the seed-mean prediction did not;
the frozen stop rule prohibited a strength or probability scan.

A separate fully-external NTU revisit completed 3 seeds × 5 subject folds on
GPU. It used the Kinetics + 25% NTU60 encoder, a fresh target head, epoch 1
head-only training, and epochs 2–15 layer4 + head training. This was a true
scope change, unlike the historical head-LR warmup in which layer4 remained
trainable from the first step. The paired single-seed mean was 0.660848 versus
0.660518 (+0.000329; required +0.002). Per-seed deltas were −0.010870,
+0.009223 and +0.002635; only 2/4/3 folds per seed jointly avoided micro and
subject-macro regression. Mean worst-user delta was −0.001517 and the largest
cell drop was −0.035461. The treatment failed five of six frozen criteria, so
the stop rule prohibited full-data training, test inference, fusion,
packaging, and submission. That result rejects progressive unfreezing only;
PKU-MMD was not part of the treatment, and the earlier permission concern was
superseded by the organiser clarification linked in the input contract.

A separate PKU-MMD bridge target confirmation then completed 15/15 GPU jobs.
The only treatment variable was the seed-matched external encoder: the control
used Kinetics + 25% NTU60, while the candidate additionally used constrained
PKU-MMD depth-only layer4 adaptation and a three-source-fold encoder soup. Both
arms used identical fresh heads, target folds, RNG seeds, 15 epochs, optimizer,
temporal-difference weight 0.10, BN policy and fixed-final evaluation.

| Frozen endpoint | Control | PKU bridge | Delta / gate |
| --- | ---: | ---: | --- |
| Mean per-seed micro | 0.669521 | 0.677866 | +0.008344; passed ≥ +0.003 |
| Mean per-seed subject-macro | 0.666864 | 0.675394 | +0.008530; passed ≥ +0.003 |
| Per-seed micro delta (2026 / 2027 / 2028) | — | — | +0.008893 / +0.006917 / +0.009223 |
| Joint micro+subject non-degraded folds | — | — | 4 / 3 / 4; failed required ≥4 for every seed |
| Mean fold-cell worst-user delta / largest cell drop | — | — | +0.015586 / −0.025339; second limit was −0.02 |
| Mean train-minus-held gap increase | — | — | −0.004006; passed |
| Three-seed logit mean (diagnostic) | 0.676877 | 0.685441 | +0.008564; worst-user 0.43125 → 0.45 |

The frozen gate passed 8/10 criteria but failed fold stability and the maximum
single-cell worst-user drop. An independent raw-logit recomputation matched
the audit exactly. The stop rule therefore prohibited full-data training,
anonymous-test inference, fusion, packaging and submission; no leaderboard
feedback was read.

### Authorized PKU bridge deployment diagnostic

The failed gate above remains the decision for the v4 promotion route. A
separate one-shot diagnostic was later authorized to answer the narrower
question: can the theoretically stronger external initialization change the
public result when its target accuracy fluctuates slightly? Before reading the
anonymous test cache, this new route fixed seed 2028, the already trained
15-epoch all-train checkpoint, uniform quantization, a 0.90 strictV3 / 0.10
bridge probability blend, OOF thresholds, package size, and a one-submission
limit. No leaderboard value was available during candidate selection.

| Frozen stage | External OOF | 90/10 blend OOF | Decision |
| --- | ---: | ---: | --- |
| FP16 reference | 2,059/3,036 = 0.678195 | 2,901/3,036 = 0.955534 | Reference only; too large for the release package |
| Uniform INT3 v1 | 364/3,036 = 0.119895 | 2,903/3,036 = 0.956192; 0 changes vs strictV3 | Signal collapse; test replay changed no submitted class, so no CSV or submission |
| Two mixed INT3/INT4 v2 variants | 355 and 365 correct | both 2,903/3,036; 0 changes vs strictV3 | Both failed train-only OOF; anonymous test remained unopened |
| Uniform INT4 v3 | 1,930/3,036 = 0.635705 | 2,900/3,036 = 0.955204; 11 changes; worst-user 0.8125 | Passed every frozen OOF and deployment gate |

The v3 package removes only the zero-release-weight thermal branch and two
redundant top-level legacy records from the nested strictV3 package. Every
executable strictV3 object remained byte-equal before and after serialization.
The earlier component sum was 99,701,322 bytes; the required single checkpoint
containing the package and YOLO is 99,978,253 bytes. Two A100 test
replays produced identical logits and CSVs; the frozen blend changed one of 405
anonymous predictions (index 36). After an independent audit, the only Kaggle
submission (`56006027`) scored **0.97512**, exactly tying canonical strictV3.
The score was recorded and the route was closed without any post-leaderboard
weight, quantizer, epoch, or seed change.

This is a completed external-data *diagnostic*, not a complete strictV3
retrain: the canonical visual, fusion, and temporal branches were not replaced,
and the external replay consumed the frozen competition cache rather than
rebuilding all raw branches. Because it tied rather than exceeded the public
baseline, the canonical 0.97512 package remains unchanged.

### NTU60 versus NTU120 scale confirmation

The next preregistered series isolated NTU scale and source coverage. All
formal models used seed 2026, five subject-wise folds, 15 fixed epochs,
layer4 plus a fresh head, frozen encoder BN, temporal-difference weight 0.10,
and uniform INT4 deployment. No fold used anonymous data.

| Frozen source / candidate | FP16 external OOF | INT4 external OOF | strictV3 90/10 INT4 blend | Decision |
| --- | ---: | ---: | ---: | --- |
| Kinetics 75% + NTU120 25% | 1,998/3,036 | 1,859/3,036 | 2,903/3,036; 10 changes | Full fit and one submission allowed |
| Kinetics 75% + NTU60 12.5% + NTU120 12.5% | 2,029/3,036 | 1,910/3,036 | 2,903/3,036; 14 changes | Package passed, but CSV duplicated ref 56006027; no submission |
| Equal target-weight soup of the two rows above | — | 1,553/3,036 | 2,902/3,036; 3 changes | Failed the preregistered 1,900-row external gate; stopped before test |
| Kinetics 75% + NTU60 25% control | 2,044/3,036 | 1,875/3,036 | 2,902/3,036; 13 changes | Failed the preregistered 1,900-row external gate; stopped before full fit/test |

The NTU120 full fit ended at 0.962121 train accuracy. Its package and YOLO
materialize as one 99,971,501-byte checkpoint; it replayed exactly twice and
changed only anonymous index 133.
Submission `56014518` scored **0.97512**. This shows that the larger NTU120
source survived deployment without hurting the public score, not that scale
alone improved it. The NTU60 control retained higher FP16 target OOF, while
the broader source mixture was more INT4-stable than NTU120 alone. The series
therefore supports treating source relevance, quantization robustness, and
data volume as separate variables.

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
