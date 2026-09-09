# Fixed-base-weight quality-gate ablation

[简体中文](README.zh-CN.md) | [Experiment index](../README.md) | [Technical report](../../../docs/TECHNICAL_REPORT.md)

This mechanism-driven exploratory ablation uses the reused current
2,889/3,036 paired T+V OOF. It asks only whether confidence-derived per-sample
quality factors add value beyond the existing base weights. It does not retrain
branches, fit parameters, inspect anonymous test data, or scan weights,
temperatures, thresholds, entropy, margins, or top-k rules.

On rows where both primary inputs are valid, the control is the released
`apply_gate`; the candidate replaces both quality factors with the same
constant. Temporal/Visual base weights remain 0.67/0.22, temperatures remain
1.1651778/1.5271352, and the Visual package-output scale remains 0.5. The 103
no-primary rows and two partial-primary rows retain their exact control logits.

## Result

| Metric | Quality gate on | Quality gate off | Delta |
| --- | ---: | ---: | ---: |
| Correct | 2,889 / 3,036 | 2,893 / 3,036 | +4 |
| Subject-macro accuracy | 0.950617 | 0.952000 | +0.001384 |
| Worst-user accuracy | 0.8125 | 0.8125 | 0 |

The candidate corrected six and broke two rows; fold nets were
`+1/0/+1/+1/+1`. It repaired five of the 53 Visual-only-top-1 errors and one of
the six Temporal-only-top-1 errors, while repairing none of the 15 joint-top-1
errors or 73 no-primary errors. It broke 2 of the 2,859 originally correct
both-primary-valid rows.

The consistency check passed: all eight changes occurred on branch-disagreement
rows, with zero changes among 2,655 rows where both branches had the same unique
top-1, and zero changes outside the both-primary-valid subset. The fixed
effective Visual weight is 0.24719; the control's sample-dependent Visual
weight ranged from 0.08435 to 0.32395.

This small positive result is insufficient for promotion. Exact two-sided
McNemar p is 0.2891 and the user-cluster bootstrap 95% interval is
`[-0.000675, 0.003083]`. Both broken rows belong to user 18, producing that
user's net -2 even though four folds gain one row and one fold is flat. The
preregistered bootstrap gate therefore fails. No release-path candidate, test
inference, submission, or more complex fusion scan follows.

This comparison has a separate score lineage from both historical strictV3
2,903/3,036 and the inherited-gate T+V diagnostic 2,909/3,036. The current
paired 2,889→2,893 result does not improve or update those figures and does not
change the released public score claim of 0.97512. The release baseline is
retained and this T+V fusion path is frozen until new evidence supports a new,
independently testable mechanism.

On 2026-09-09, the user explicitly overrode the no-submission action after the
freeze. A deployment-matched gate-off CSV was generated without anonymous
labels and submitted as Kaggle ref `56117280`; it scored `0.97512`, tying the
existing release. This external action does not retroactively pass the
preregistered gate, promote the candidate, or reopen fusion tuning. Its inputs,
predictions and submission record are preserved in
[`kaggle_submission/`](kaggle_submission/).

The archive retains [preregistration.json](preregistration.json), full paired
[predictions.npz](predictions.npz), [input and output hashes](provenance.json),
[metrics.json](metrics.json), [changes.csv](changes.csv), and the frozen
[decision](decision.json). The reproducible evaluator is
[`quality_gate_ablation.py`](../../../yolo_r2plus1d/strict_v3/evaluation/quality_gate_ablation.py).
