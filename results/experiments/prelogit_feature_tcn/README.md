# Pre-logit feature TCN

[简体中文](README.zh-CN.md) | [Experiment index](../README.md) | [Technical report](../../../docs/TECHNICAL_REPORT.md)

This paired five-fold OOF experiment tests one structural question: does the
released DSTFormer retain temporal evidence before its final 40-class frame
classifier that a TCN cannot recover from frame logits?

The control is `mean(frame_logits) + TCN(frame_logits)`. The candidate keeps
the exact same static mean and changes only the TCN input to
`PCA40(pre_fc2_features)`. Each PCA is fit on the outer-training users' frames
and then frozen. The two inputs come from the same DSTFormer forward pass; the
exported control logits exactly match the earlier cache (maximum difference
zero, identical SHA256). TCN architecture, parameter count, 16-frame sampling,
seed, five-epoch budget, Visual OOF and release T+V fusion contract are fixed.

## Result

| Measurement | Logit TCN | Feature TCN | Delta |
| --- | ---: | ---: | ---: |
| Temporal correct | 2,847 / 3,036 | 2,839 / 3,036 | -8 |
| Frozen T+V correct | 2,889 / 3,036 | 2,891 / 3,036 | +2 |
| T+V subject-macro accuracy | 0.950617 | 0.951393 | +0.000777 |
| T+V both-primary-valid correct | 2,859 / 2,931 | 2,861 / 2,931 | +2 |
| T+V no-primary correct | 30 / 103 | 30 / 103 | 0 |

The final T+V prediction corrected six rows and broke four. Fold net changes
were `A 0, B -1, C +2, D 0, E +1`, so the preregistered gate technically
passes. The gain is not robust evidence: only ten predictions changed, exact
two-sided McNemar p is 0.7539, and a user-cluster bootstrap 95% interval for
accuracy delta is `[-0.000983, 0.002628]`.

The mechanism is mixed rather than confirmed. The candidate worsens Temporal
alone in three folds and is flat in two; its small final gain exists only in
interaction with frozen Visual. Changes span six users, but two Wash-face
corrections for user 8 are offset by two Wipe-hands failures for the same user,
and Read-documents/Turn-pages also changes in both directions. We therefore do
not promote, full-fit, submit, or scan PCA widths/attention variants. The
evidence lowers the priority of "40-D logits are the main temporal bottleneck."

Machine-readable evidence is in [metrics.json](metrics.json),
[stability.json](stability.json), [preregistration.json](preregistration.json),
[feature_export.json](feature_export.json), and
[input_subset_decomposition.json](input_subset_decomposition.json).

## Reproduce the inputs

Run from the repository root with zsh. The output directory must not already
exist, which prevents an earlier PCA cache from being silently reused.

```zsh
.conda/envs/cuhkx/bin/python -m \
  yolo_r2plus1d.strict_v3.data.prelogit_features \
  --package checkpoints/strict_v3/model.pt \
  --output-dir runs/experiments/prelogit_feature_tcn_v1/inputs \
  --reference-logits \
    runs/experiments/skeleton_retargeting_v2/inputs/original_frame_logits.npy \
  --device cuda:0
```

Train each fold twice with
`yolo_r2plus1d.strict_v3.training.temporal`: the control receives only
`--logits .../frame_logits.npy`; the candidate additionally receives its
`--temporal-input .../foldX_pca40.npy`. Both use `--kind tcn --epochs 5
--scheduler-epochs 30 --seed 2026 --select-last --defer-val-metrics` and the
same held-user fold listed in [preregistration.json](preregistration.json).
