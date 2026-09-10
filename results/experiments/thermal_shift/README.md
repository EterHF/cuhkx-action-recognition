# Thermal temporal-shift ablation

[简体中文](README.zh-CN.md) | [Thermal control](../thermal_fallback/README.md)

The only change is a [TSM](https://arxiv.org/abs/1811.08383) before the first convolution
of every ResNet-18 residual block: 1/8 of channels move in each temporal direction,
with zero padding and no exchange across clips. No learned parameters are added.
The cache, augmentation, 15 epochs, three seeds, subject folds, and fallback policy remain fixed.

Net fallback corrections are +11/+5/+8 versus the control's +10/+8/+13, totaling seven
fewer correct predictions. Seed 2026's user-cluster interval still has a zero lower bound.
Both the uncertainty gate and additional improvement-over-TSN gate fail. No full fit,
anonymous inference, or submission was performed.

This rejects this particular ablation, not temporal modeling in general. It retains
the control's four-frame training and eight-frame evaluation sampling. No shift ratio,
placement, sampling, or epoch sweep followed. See [metrics](metrics.json),
[preregistration](preregistration.json), and [provenance](provenance.json).

To reproduce, share the control input cache, add `--temporal-shift` to the thermal suite,
and evaluate with `--temporal-shift --control-output results/experiments/thermal_fallback`.
