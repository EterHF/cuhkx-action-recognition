# StrictV3 consensus experiments

[English](README.md) | [简体中文](README.zh-CN.md) | [Repository home](../../README.md) | [Technical report](../../docs/TECHNICAL_REPORT.md)

These candidates were recovered from the audited work in Codex thread
`01a04b41-e2ed-79c3-9fce-20eec40a3c73` and replayed with the current public
pipeline. They are deliberately separate from `checkpoints/strict_v3/model.pt`:
the `main` baseline remains the byte-reproducible 0.97512 release.

| Candidate | OOF | Delta | Fold gate | Single checkpoint | Test changes | Kaggle |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| [`sched30_consensus`](sched30_consensus/) | 0.960474 | +0.004282 | 5/5 non-degrading | 98,873,941 B | 3 | not submitted |
| [`temporal_pool_consensus`](temporal_pool_consensus/) | 0.959816 | +0.003623 | 5/5 non-degrading | 96,936,797 B | 12 | not submitted |

The first candidate averages probabilities from strictV3 and a fixed
three-seed sched30 temporal release. The second uses a majority vote over
motion-energy, top-2-frame and top-4-frame temporal pooling while sharing the
large model states.

The 0.960474 row is the frozen audited artifact. A fresh CPU retrain on
2026-09-03 produced 0.959486 after recalibration because a numerical grid tie
selected an adjacent weight. That rerun is documented as supporting evidence,
not as a replacement for this frozen candidate.

Audit both frozen candidates:

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.experiments.audit
```

Reproduce one candidate from raw test data and validate its frozen hash:

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.bundle \
  --model checkpoints/experiments/sched30_consensus.pt \
  --output /tmp/inference_bundle.pt
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --bundle /tmp/inference_bundle.pt \
  --output .cache/sched30_consensus.csv
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.experiments.audit \
  sched30_consensus --replayed-csv .cache/sched30_consensus.csv
```

No candidate is promoted or submitted based on anonymous-test inspection.
Promotion requires the frozen OOF gates, exact raw replay, the single-checkpoint 100 MB package
gate and an explicitly authorized Kaggle submission.
