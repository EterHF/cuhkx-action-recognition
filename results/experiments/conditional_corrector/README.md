# Temporal-conditioned Visual corrector

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

The implemented 77k-parameter head consumes the Visual pre-classifier feature
and frozen Temporal logits, then adds a 40-way residual to the existing fused
logits. Only its final layer is zero initialized, so the initial model is
exactly the baseline while the hidden projection remains trainable.

Training minimizes final-prediction cross-entropy plus
`KL(p0 || p_new)` on training rows where the baseline is both correct and at
least 0.8 confident. Evaluation reports corrected, broken and net rows; the
corrector is not required to classify independently.

The strict evaluation is now complete. It uses fixed external-only encoders
(PKU-MMD R(2+1)D-34 for Visual and NTU DSTFormer for skeleton Temporal), then
fits fresh 40-class heads in 20 outer-by-inner runs. Every meta-training row is
inner cross-fitted, every upstream head excludes the outer-held users, and all
checkpoints use a fixed final epoch. No anonymous-test input was accessed.

On 3,036 nested-OOF rows, the frozen T+V baseline rose from 1,201 correct
(`0.395586`) to 1,465 (`0.482543`): 438 rows were corrected, 174 were broken,
and the net correction was +264. Fold A-E net changes were
`+46/+58/+48/+66/+46`; all 18 users also improved. Subject-macro accuracy rose
from `0.391962` to `0.479934`, worst-user accuracy from `0.1625` to `0.29375`,
and the user-clustered 95% interval for accuracy change was
`[+0.07310,+0.10375]`. A second identical run reproduced both OOF arrays
byte-for-byte. Fixed-seed replications at seeds 2026/2027/2028 produced net
corrections +264/+261/+276, with all 15 fold-seed pairs positive.

This is evidence that conditional residual correction works under strict
isolation, but it is not a strictV3 deployment estimate. The external-only
proxy is much weaker than strictV3, and its Visual branch (1,310 correct) is
stronger than its Temporal branch (902), so the frozen strictV3
Temporal-dominant fusion weights are mismatched. The corrector still exceeded
Visual alone by 155 rows, but no full-fit, test inference, or Kaggle submission
is authorized from this result.

A separate deployment-aligned screen rebuilt uniform-5-bit Visual OOF logits
from the original fold checkpoints using the release single-view preprocessing.
Its frozen T+V baseline scored 2,890/3,036 (`0.951910`) and reproduced the
previous T+V test predictions exactly. The 512-D corrector fell by 24 rows.
The one preregistered coordinate-alignment change—using 40-D Visual class
logits—still fell by 11 rows (32 corrected, 43 broken), with fold nets
`+6/+9/+6/0/-32`. Subject-macro fell from `0.951297` to `0.947341`, and
worst-user accuracy from `0.8125` to `0.625`.

The failure is highly localized rather than harmless noise: 30 correct class-36
examples from user 5 were changed to class 10. This class-by-subject bias
dominates Fold E and contradicts the desired unseen-user generalization. Both
preregistered deployment variants therefore failed promotion; full-fit and the
Kaggle submission were deliberately not run. Exact machine-readable results are
in [`deployment_screen.json`](deployment_screen.json).
