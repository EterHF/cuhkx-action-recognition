# Mild skeleton bone retargeting

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

The first absolute-length reconstruction failed geometry QC before any model
training because its maximum normalized displacement was 2.474. The accepted
v2 applies one clip-fixed ±5% scale per bilateral bone group while preserving
per-frame projected-length variation, direction, root trajectory, confidence,
and missing frames. Its mean/max displacement is 0.0153/0.1460.

Using a deterministic 50/50 original/retargeted training assignment, with
original-only held-user validation, changed Temporal from 2,847 to 2,846 and
T+V from 2,889 to 2,888. It corrected no rows and broke one (fold nets
`0/-1/0/0/0`). The transform is stable but provides no positive generalization
evidence, so no magnitude scan followed.
