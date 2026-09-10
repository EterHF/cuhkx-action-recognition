# Thermal fallback, 2026-09-10

[简体中文](README.zh-CN.md) | [Experiment index](../README.md)

All 103 training clips missing both Visual and Skeleton have Thermal, while none have
IMU or Radar. This experiment trains an independent ImageNet ResNet-18 temporal segment
classifier and uses it only when both primary inputs are absent and Thermal is present.
Every other baseline logit remains bitwise unchanged.

The [TSN-inspired](https://arxiv.org/abs/1608.00859) recipe uses 16 cached full-field
160×160 frames, four sampled training frames, eight evaluation frames, frozen BN running
statistics, and 15 fixed epochs. Three seeds each run all five subject folds. Each fold
starts from original ImageNet weights and evaluates its final FP16-stored state once.
Deployment seed 2026 was specified before held results were inspected.

| Seed | Correct among 103 missing-primary clips (baseline: 30) | Net corrections |
| --- | ---: | ---: |
| 2026 | 40 | +10 |
| 2027 | 38 | +8 |
| 2028 | 43 | +13 |

All 15 seed/fold combinations are non-degrading and subject macro accuracy improves.
However, nine of seed 2026's ten net corrections come from user 5. Its user-cluster
bootstrap 95% interval for micro-accuracy gain is `[0, 0.009521]`, failing the registered
strictly positive lower-bound gate. No full fit, anonymous inference, or Kaggle submission
was performed. Selecting seed 2028 after seeing its interval would violate the protocol.

The report separately retains historical release-strict (2,903/3,036) and current paired
T+V (2,889/3,036) baselines; the same net corrections apply to both, but their absolute
scores are not interchangeable. Upstream target-label exposure in the baseline encoders
means the combined scores are not unbiased private-leaderboard estimates. Thermal itself
achieves only about 37% held-subject accuracy on valid thermal inputs.

Each fold checkpoint is 22,444,867 bytes. No complete deployment bundle was built or
verified. The 0.97512 release remains unchanged. See the
[Chinese reproduction commands](README.zh-CN.md#复现), [preregistration](preregistration.json),
[metrics](metrics.json), [predictions](predictions.npz), and [provenance](provenance.json).
Related paired ablations are [temporal shift](../thermal_shift/README.md) and
[person cropping](../thermal_crop/README.md); reused subjects are not independent confirmations.
