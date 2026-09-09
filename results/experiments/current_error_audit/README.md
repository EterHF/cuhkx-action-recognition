# Current T+V error and input-evidence audit

[简体中文](README.zh-CN.md) | [Experiment index](../README.md) | [Technical report](../../../docs/TECHNICAL_REPORT.md)

This audit uses the current paired 2,889/3,036 T+V OOF baseline. It does not
reuse the older 2,903-row release error list, retrain a model, inspect anonymous
test data, tune fusion, or combine historical candidate predictions.

Of 147 errors, 73 are among the 103 clips with neither primary input. The other
74 comprise 6 where only Temporal top-1 is correct, 53 where only Visual top-1
is correct, and 15 where both top-1 predictions are wrong. In all 15 jointly
wrong rows, the true class is Temporal rank 2; Visual ranks it 2 in 11 rows, 3
in one, and 13/25/36 in one row each. Thus the remaining valid-input errors are
mostly failures to preserve already available branch evidence, not cases where
both branches rank the truth far down. This coverage is descriptive and is not
a soft-fusion accuracy ceiling or authorization to increase Visual weight.

Across the six comparable recent candidates, only 10 baseline errors were ever
corrected and 137 persisted. The persistent set contains all 73 no-primary
errors, 6 Temporal-only-correct errors, 45 Visual-only-correct errors, and 13
joint top-1 errors. The per-row evidence is in [errors.csv](errors.csv); it must
not be used as an oracle ensemble or corrector training list.

A blinded evidence review sampled one row from each of 10 independent
persistent joint-error clusters and paired it with a same-class, different-user
correct control. Every raw Depth_Color frame, the selected 16 uncropped frames,
the union-cropped 128×128 Depth/IR cache, and skeleton were reviewed side by
side before opening labels or predictions. No clear sampling or spatial loss
was observed. Suspected skeleton anomalies occurred equally in errors and
controls (2/10 each), and source frame counts aligned for all modalities in all
20 clips. This small descriptive audit therefore authorizes no sampling,
local-view, resolution, or skeleton-repair training.

See [metrics.json](metrics.json), [review_summary.json](review_summary.json),
[blind_observations.csv](blind_observations.csv), and
[review_answer_key.csv](review_answer_key.csv).

The reproducible analyzers are
[`current_error_audit.py`](../../../yolo_r2plus1d/strict_v3/evaluation/current_error_audit.py)
and
[`build_evidence_review.py`](../../../yolo_r2plus1d/strict_v3/evaluation/build_evidence_review.py).
Generated contact sheets remain under `runs/` rather than being committed; the
review summary, blinded observations and answer key are retained here.
