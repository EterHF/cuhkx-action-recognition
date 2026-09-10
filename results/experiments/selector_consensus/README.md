# Unanimous bounded-selector ensemble

[English](README.md) | [简体中文](README.zh-CN.md)

Three fixed bounded selectors must unanimously choose the same alternative before changing baseline;
otherwise baseline logits remain exact. Deployment uses seeds2026/2027/2028. New seeds2029–2031
were fitted after locking unanimity to replicate initialization stability.

Both triplets add 26 correct rows to the deployment-aligned 2,890/3,036 baseline, reaching
2,916/3,036 (96.0474%). Fold nets are +6/+13/+6/0/+1 for both. Subject macro improves and
worst-user accuracy is preserved. The user-cluster 95% interval is +0.3560 to +1.4459 percentage
points. Deployment corrects35/breaks9; replication corrects37/breaks11.

Full fitting uses 264 eligible branch-disagreement training examples per head. Raw training
availability exactly reproduces the OOF mask. The single checkpoint containing every weight,
including YOLO, is 50,730,395 bytes. See `deployment.json` for two raw replay and Kaggle results.

This is exploratory adaptive design following historical error inspection. New seeds are not an
independent dataset, and upstream target pretraining remains a limitation. Neither these OOF gains
nor intervals establish private-LB improvement. Canonical release files remain reproducible.
Raw inference entry point: `python -m yolo_r2plus1d.strict_v3.release.replay --bundle <candidate_bundle> --output <csv>`.

Evidence: [preregistration.json](preregistration.json), [metrics.json](metrics.json), [predictions.npz](predictions.npz).

Public validation: submission **56145116** on2026-09-10 scored **0.96019**, below the existing **0.97512** best. Promotion is rejected; this route is frozen without leaderboard-driven tuning/resubmission. Canonical checkpoint/CSV are unchanged; private score is unavailable.
