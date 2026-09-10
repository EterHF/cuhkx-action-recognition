# Local/world IMU ablation

[English](README.md) | [简体中文](README.zh-CN.md)

Thirty fixed-epoch models compare equal-capacity local-coordinate and local+quaternion-world IMU views
across three seeds and five held-user folds. Each has 126,632 parameters and a 273,637-byte checkpoint.
The parser uses five anatomical locations, physical units and normalized within-clip time; it excludes
absolute time, device identity and auxiliary metadata. Rotation direction passed a 13,053-reading gravity audit.

Mean standalone accuracy was 30.8739% for world views versus 31.4112% for local views. The registered
10% probability contribution yielded canonical net corrections +2/+1/+1 and current paired T+V
+1/+2/0. The locked gates failed; no full-fit, anonymous inference, weight scan or submission followed.
Models, logs and cache are in `runs/experiments/imu_global_v1/`. Upstream baseline target pretraining
means fusion OOF remains engineering evidence, not unbiased unseen-user validation.

References: [InceptionTime](https://arxiv.org/abs/1909.04939), [FLOW](https://arxiv.org/abs/2406.18569).

Evidence: [preregistration.json](preregistration.json), [metrics.json](metrics.json), [predictions.npz](predictions.npz).
