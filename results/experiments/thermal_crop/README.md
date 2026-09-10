# Thermal person-crop ablation

[简体中文](README.zh-CN.md) | [Thermal control](../thermal_fallback/README.md)

A label-independent audit of every twentieth training row found person detections in
133/142 valid thermal clips (93.66%), with crop area ranging from 17.3% to 100% of the
original frame. This motivated one paired input change: perform the existing YOLO11n
union crop in the thermal camera's own coordinates.

The detector settings are inherited from Visual: eight probes, confidence 0.25,
margin 1.4, minimum side 0.35, and full-field fallback on no detection. All other TSN
training settings remain fixed, with no temporal shift. Full training coverage is
2,764/2,891 valid thermal clips (95.61%).

Net fallback corrections are +3/+9/+3 across seeds, versus +10/+8/+13 for full-field
TSN: sixteen fewer correct predictions in aggregate. Seed 2026's user-cluster accuracy
gain interval is `[0, 0.002009]`. The crop fails both the uncertainty and paired
improvement gates. No full fit, anonymous inference, or submission was performed,
and detector thresholds, margins, or crop modes were not swept afterward.

The existing detector can be shared, but no complete deployment bundle was built.
See [reproduction commands](README.zh-CN.md#复现), [metrics](metrics.json),
[predictions](predictions.npz), [preregistration](preregistration.json),
and [provenance](provenance.json). Models, logs, input cache, windows, and the suite
receipt remain under `runs/experiments/thermal_crop_v1/`.
