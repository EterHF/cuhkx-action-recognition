# Final submission candidates

[English](README.md) | [简体中文](README.zh-CN.md) | [Project status](../../docs/PROJECT_STATUS.md)

**Website selection has not been completed or verified.** The current Kaggle OAuth
login can read/download submissions through the ordinary API, but the website's
final-selection endpoint returns `401 Unauthenticated`. Recommendations are not
reported as completed selections. In the authenticated
[submissions page](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/submissions),
select the following pair and verify exactly these two IDs.

| Submission | Model | Submitted UTC | Filename | Public score | All weights |
| --- | --- | --- | --- | ---: | ---: |
| **55712568** | Canonical strictV3 | 2026-08-23 10:35:47 | submission1.csv | **0.97512** | 69,805,793 B |
| **56036959** | Inherited-gate Temporal + Visual | 2026-09-05 16:34:50 | submission.csv | **0.97512** | 50,733,469 B |

Both CSVs were downloaded from Kaggle and match the repository byte-for-byte.
They differ on **3/405** predictions. T+V removes weak Fusion without fitting new
parameters; the separate bundle removes inactive weights. This offers a simpler,
reproducible alternative tied at the best known public score, not a guarantee of
better private performance. The early 2,909-correct T+V OOF had a Visual-scale
mismatch and is not used to justify selection. The new selector scored 0.96019;
gate-off tied publicly but failed its original OOF gates, so neither is preferred.

## Artifacts and checks

* Canonical CSV: [`submission.csv`](../strict_v3/submission.csv)
* T+V CSV: [`submission.csv`](../experiments/temporal_visual_inherited/submission.csv)
* Canonical bundle: [`submission_bundle.pt`](../../checkpoints/strict_v3/submission_bundle.pt)
* T+V bundle: [`temporal_visual_inherited_bundle.pt`](../../checkpoints/experiments/temporal_visual_inherited_bundle.pt)
* [`selection.json`](selection.json): hashes, raw replay checks and the website's 401 status.

Each bundle contains all inference weights in one file strictly below 100,000,000
bytes. Two independent raw T+V replays have identical CSV bytes and branch logits,
matching the downloaded historical submission. No predictions were edited and
no new submission was made for final selection.

Replay the second candidate from the repository root:

```bash
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --bundle checkpoints/experiments/temporal_visual_inherited_bundle.pt \
  --output runs/final_selection_replay/submission.csv
cmp runs/final_selection_replay/submission.csv \
  results/experiments/temporal_visual_inherited/submission.csv
```

For canonical strictV3, use the root README's verification and replay commands.
