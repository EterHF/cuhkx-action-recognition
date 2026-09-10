# Availability-aware residual corrector

[English](README.md) | [简体中文](README.zh-CN.md)

The old semantic residual head broke 30 previously correct missing-primary rows in user5. This experiment
keeps its 60-epoch CE+KL recipe but restricts training and inference to both-primary-valid rows.
Unavailable baseline logits remain exact; routing uses no class, user or anonymous identifier.

Fifteen models yield +17/+20/+17 net corrections over the deployment-aligned 2,890/3,036 baseline,
but Fold E loses 3/3/1 rows. The deployment seed user-cluster 95% interval includes zero.
The standalone recipe is rejected without full-fit, anonymous inference or submission. Subsequent
error diagnosis motivated the bounded selector and is explicitly adaptive use of held evidence.

Evidence: [preregistration.json](preregistration.json), [metrics.json](metrics.json), [predictions.npz](predictions.npz).
