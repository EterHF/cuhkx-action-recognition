# StrictV3 consensus experiments

These candidates were recovered from the audited work in Codex thread
`01a04b41-e2ed-79c3-9fce-20eec40a3c73` and replayed with the current public
pipeline. They are deliberately separate from `checkpoints/strict_v3/model.pt`:
the `main` baseline remains the byte-reproducible 0.97512 release.

| Candidate | OOF | Delta | Fold gate | Package + YOLO | Test changes | Kaggle |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `sched30_consensus` | 0.960474 | +0.004282 | 5/5 non-degrading | 98,911,347 B | 3 | not submitted |
| `temporal_pool_consensus` | 0.959816 | +0.003623 | 5/5 non-degrading | 96,964,489 B | 12 | not submitted |

The first candidate averages probabilities from strictV3 and a fixed
three-seed sched30 temporal release. The second uses a majority vote over
motion-energy, top-2-frame and top-4-frame temporal pooling while sharing the
large model states.

Audit both frozen candidates:

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.experiments.audit
```

Reproduce one candidate from raw test data and validate its frozen hash:

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --package checkpoints/experiments/sched30_consensus.pt \
  --output .cache/sched30_consensus.csv
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.experiments.audit \
  sched30_consensus --replayed-csv .cache/sched30_consensus.csv
```

No candidate is promoted or submitted based on anonymous-test inspection.
Promotion requires the frozen OOF gates, exact raw replay, the 100 MB package
gate and an explicitly authorized Kaggle submission.
