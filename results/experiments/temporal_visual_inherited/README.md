# strictV3 Temporal + Visual inherited-gate optimization

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

This candidate removes the weak legacy Fusion branch without fitting any new
parameter. All strictV3 fold/full temperatures, Temporal/Visual base weights,
and confidence-quality gating remain frozen; only the Fusion base weight is set
to zero and the existing gate normalizes the remaining branches.

Subject-wise OOF improved from 2,903/3,036 (`0.956192`) to 2,909/3,036
(`0.958169`). Fold deltas A–E were `0/0/+5/+1/0`; macro recall rose from
0.953136 to 0.956076, subject-macro accuracy from 0.955794 to 0.957954, and
worst-user remained 0.8125. This is stronger than the rejected exact-equal
blend because Temporal remains dominant: the mean effective Visual test weight
is 0.1653 after confidence gating.

The unchanged 69,805,793-byte strictV3 bundle is reused. Test predictions cover
all 40 classes and differ from strictV3 on only 3/405 rows. Kaggle ref
`56036959` was submitted after the all-fold gate passed and scored `0.97512`,
tying strictV3. It is retained as a simpler private-LB candidate; the public tie
is not used for further weight search.
