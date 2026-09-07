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

Training has deliberately not started. A valid five-fold evaluation needs 20
outer-by-inner upstream runs so every corrector-training prediction is
cross-fitted inside its outer-train split and every upstream model also excludes
the outer-held users. Those features/logits are absent. The trainer rejects an
ordinary global OOF manifest rather than silently reporting leakage-biased
stacking results.
