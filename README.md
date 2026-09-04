# CUHK-X Small Model Track — strictV3

[English](README.md) | [简体中文](README.zh-CN.md) | [Documentation](docs/README.md)

The cleaned strictV3 baseline for the CUHK-X Small Model Track (UbiComp / ISWC
2026). The released package reproduces the canonical **0.97512** submission
(rank 3, submission `55712568`) byte-for-byte from raw test data. The deployable
package occupies 69.82 MB including the YOLO11n detector.

Historical experiments and rejected directions are intentionally excluded
from the main code tree. Their methods and outcomes are preserved in the
[technical report](docs/TECHNICAL_REPORT.md) ([中文版](docs/TECHNICAL_REPORT.zh-CN.md)).

## Repository layout

```text
checkpoints/strict_v3/       released model and detector (Git LFS)
results/strict_v3/           OOF/test logits, metrics and canonical CSV
yolo_r2plus1d/strict_v3/
├── data/                    deterministic indexing and cache builders
├── models/                  R(2+1)D, DSTFormer, fusion and optimizer code
├── training/                visual, skeleton, temporal and fusion training
├── inference/               branch-level inference
├── release/                 raw replay, fusion contract and verification
└── cli/                     guarded Kaggle submission helper
tests/                       fast release and data-contract tests
docs/TECHNICAL_REPORT.md     experiment history and design rationale
```

## Git navigation

| Destination | Link |
| --- | --- |
| strictV3 source | [`yolo_r2plus1d/strict_v3/`](yolo_r2plus1d/strict_v3/) |
| Training entry points | [`training/`](yolo_r2plus1d/strict_v3/training/) |
| Release and verification | [`release/`](yolo_r2plus1d/strict_v3/release/) |
| Tests | [`tests/`](tests/) |
| Released checkpoints | [`checkpoints/strict_v3/`](checkpoints/strict_v3/) |
| Results and manifests | [`results/strict_v3/`](results/strict_v3/) |
| Data layout | [`data/README.md`](data/README.md) |
| Retraining guide | [`docs/RETRAINING.md`](docs/RETRAINING.md) |
| Technical report | [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) |

Generated caches, datasets and training runs are not versioned.

NTU RGB+D and PKU-MMD are external research inputs, not strictV3 release
dependencies. Their archives and source-derived experimental checkpoints are
excluded from this repository; the canonical 0.97512 verification and replay
do not read `data/external/`. The competition host explicitly permits public
external datasets and pretrained models, including NTU RGB+D, provided that
access and use are disclosed in the final writeup. This permission to train is
separate from redistribution: upstream archives remain unversioned and subject
to their providers' terms. See the
[official clarification](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/724404),
the [data boundary](data/README.md#external-research-datasets), and the technical
report before attempting external-data research.

Two OOF-qualified consensus candidates are maintained on the
`experiment/strictv3-consensus` branch. Their frozen packages, raw replay
hashes and audit command are documented in
[`results/experiments/README.md`](results/experiments/README.md); neither has
been submitted to Kaggle.

## Installation

```bash
git lfs install
git lfs pull
conda env create -p .conda/envs/cuhkx -f environment.yml
```

All commands below run from the repository root.

## Verify the published result

The fast CPU check validates every published hash, the package safety
contract, the 100 MB limit, the saved OOF score and the submission generated
from the saved test logits:

```bash
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify
```

Expected headline values:

```text
OOF accuracy: 0.9561923583662714
submission rows/classes: 405 / 40
model + detector: 69,819,764 bytes
```

## Replay inference from raw test data

Place the competition test tree at:

```text
data/Small-Model-Track/Testing/test_file/{test.csv,sample_submission.csv}
data/processed/test/small_model_track_test/SM_test_*/
```

Then run the complete YOLO crop, visual-cache, skeleton-cache and three-branch
inference pipeline:

```bash
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --output results/strict_v3/reproduced_submission.csv

.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify \
  --replayed-csv results/strict_v3/reproduced_submission.csv
```

The replay never reads test labels, test-derived statistics, timestamps or
user identities. It deterministically reproduces the canonical CSV hash
`e2509491…` byte-for-byte. GPU inference is recommended; use `--device cpu`
only for a slow functional replay.

## Training

Raw replay and model retraining are separate checks. The command above verifies
the published checkpoint; it does not retrain it. The exact input boundary,
folds, commands and acceptance criteria for retraining are documented in
[`docs/RETRAINING.md`](docs/RETRAINING.md).

Training caches are generated by the modules in `data/`; training entry points
live in `training/`. The temporal suites have a single auditable runner:

```bash
# Five strictV3 folds plus the full epoch-5 model
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.temporal_suite \
  --recipe strict --frame-logits .cache/strict_v3/train_frame_logits.npy \
  --test-frame-logits .cache/strict_v3/test_frame_logits.npy \
  --metadata results/strict_v3/metadata.npz --output runs/strict_v3/temporal

# Three seeds, five folds and three full models for the sched30 experiment
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.temporal_suite \
  --recipe sched30 --frame-logits .cache/strict_v3/train_frame_logits.npy \
  --test-frame-logits .cache/strict_v3/test_frame_logits.npy \
  --metadata results/strict_v3/metadata.npz --output runs/experiments/sched30

# Visual R(2+1)D baseline
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.base --help

# Released visual-head family
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.public_finetune --help

# Skeleton/visual fusion and temporal residual head
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.fusion --help
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.temporal --help
```

For a train-only confirmatory run, omit `--test-frame-logits` and add
`--oof-only`. The suite then skips every full-data job, refuses a test-logit
argument, and records `test_data_loaded: false` in its receipt.

Subject-wise folds and frozen release hyperparameters are recorded in
`results/strict_v3/release_manifest.json`. Do not use leaderboard feedback or
anonymous-test attributes for model selection.

## Retraining status (2026-09-04)

The temporal, visual-head and fusion fold jobs were rerun from the declared
bootstrap boundary. Temporal reproduced all five historical fold accuracies;
fusion reproduced all five historical best accuracies; visual-head results
remained within three validation rows of the historical folds. The separately
retrained `sched30` candidate reached 0.959486 OOF after recalibration, so it
has **not** replaced the frozen strictV3 release or the separately audited
0.960474 candidate package. Exact commands, fold results and the remaining
full-deployment acceptance gates are in
[`docs/RETRAINING.md`](docs/RETRAINING.md).

Three additional preregistered, train-only checks were rejected. Resampling the
same temporal augmentation each epoch left the three-seed prediction mean at
0.937747. Uniformly averaging the epoch 3–5 TCN weights reached 0.938076
(+1/3,036 row), but its fixed nested 50/50 release candidate remained exactly
0.960474 with zero changed predictions. A fixed 0.25 same-class cross-user
temporal-residual mix changed the logits but moved none of 3,036 top-1
decisions. None of the checks opened test data, trained full-data models, or
changed the published package.

A separate fully-external NTU check ran 3 seeds × 5 subject folds on GPU. A
true epoch-1 head-only stage followed by layer4 + head adaptation improved the
paired single-seed mean by only 0.000329 (required 0.002), regressed seed 2026
by 0.010870, and reduced worst-user robustness. It failed five of six frozen
criteria, so no full model, test inference, fusion, package or submission was
created.

After correcting the external-data policy record, a fresh seed-matched
Kinetics → NTU60 → PKU-MMD bridge confirmation completed another 3 seeds × 5
subject folds. With the target recipe held fixed, mean micro rose from 0.669521
to 0.677866 (+0.008344) and subject-macro rose by 0.008530; every seed improved.
The diagnostic three-seed logit mean reached 0.685441 versus 0.676877. This is
real external-data gain, but the frozen promotion gate remained stricter: seed
2027 had only 3/5 jointly non-degraded folds (required 4/5 for every seed), and
one cell's worst-user score dropped 0.025339 (limit 0.02). The route therefore
stopped under its original promotion rule.

A separately preregistered, explicitly authorized one-shot deployment
diagnostic then reused the fixed seed-2028, 15-epoch full-data checkpoint. INT3
and mixed INT3/INT4 packages collapsed during train-only OOF audits; the final
uniform-INT4 member retained 1,930/3,036 external OOF rows. Its frozen 90/10
blend with strictV3 scored 2,900/3,036 OOF, changed 11 OOF decisions, preserved
the 0.8125 worst-user floor, and occupied 99,701,322 bytes with YOLO. Two local
GPU replays were identical and changed one anonymous prediction. The single
Kaggle confirmation, ref `56006027`, scored **0.97512**: an exact tie, not an
improvement. No leaderboard retuning followed, so canonical strictV3 remains
the public main baseline and the external-data diagnostic is report-only.

The follow-up NTU scale study retrained fixed seed-2026 target models from
Kinetics+NTU120, an equal NTU60/NTU120 source mixture, and the NTU60 control.
The deployable NTU120 member passed its INT4 train-only gate: the strictV3
90/10 blend retained 2,903/3,036 OOF rows, preserved the 0.8125 worst-user
floor, and changed 10 OOF predictions. Its two package replays were identical
and changed anonymous index 133; Kaggle submission `56014518` scored
**0.97512**, again tying rather than improving strictV3.

The source mixture improved external INT4 OOF from 1,859 to 1,910 correct but
produced a CSV byte-identical to the already submitted PKU diagnostic, so it
was not resubmitted. A target-weight soup failed its external OOF gate, and the
retrained NTU60 control failed its frozen INT4 external threshold (1,875 <
1,900); neither accessed anonymous data. Three daily submissions remained,
but no additional candidate satisfied both the evidence and novelty gates.

## License

Code is released under the [MIT License](LICENSE). Competition data and
pretrained weights remain subject to their respective upstream terms.
