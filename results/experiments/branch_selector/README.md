# Bounded binary branch selector

[English](README.md) | [简体中文](README.zh-CN.md)

A 6→16→2 MLP chooses only Temporal or Visual top1 on both-primary-valid disagreements with baseline
confidence below .8. That threshold is inherited from the original corrector protection rule and was
not scanned. Training uses disagreements whose label is represented by either choice, 100 fixed
AdamW .01 epochs, and no early stopping.

The original three seeds improve net correctness by +23/+25/+27, with positive user-cluster 95%
intervals. Seeds2027/2028 nevertheless lose one Fold-D row, so individual models fail promotion.
The separately registered unanimity experiment uses all three seeds; new seeds2029–2031 replicate
initialization stability. No favorable seed is selected, and no standalone selector is submitted.

Evidence: [preregistration.json](preregistration.json), [metrics.json](metrics.json), [predictions.npz](predictions.npz).
