# Full-fit public sensor ensemble

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

All 3,036 labelled clips were used to retrain three independent branches:
Depth and IR Omnivore Swin-T from the official public initialization, and the
native-rate skeleton model from random initialization. Sensor training used
random intensity scale/offset augmentation; skeleton training used valid-frame
noise and branch dropout. No historical project checkpoint was loaded.

Unaugmented in-sample accuracies were 0.931818 (Depth), 0.940711 (IR), and
0.833663 (skeleton). Their fixed equal-logit fp32 ensemble reached 0.950264
(2,885/3,036); the deployment-aligned mixed-int8 bundle reached 0.950593
(2,886/3,036). These are training accuracies, not generalisation estimates.

The single deployment bundle is 57,925,224 bytes with SHA-256 `df994402…`.
It uses per-output-channel int8 for ordinary backbone Conv/Linear weights and
fp16 for sensitive sensor layers and the skeleton branch. Test inference
produced 405 predictions with 39/40 natural class coverage (class 25 absent).
No prediction was changed to manufacture class coverage.

The frozen submission SHA-256 is `fe85be36…`. Kaggle submission 56035438 scored
**0.58706** publicly. This agrees with the earlier ~0.598 reused-fold research
diagnostic and is far below strictV3's 0.97512. The candidate is rejected; the
score was recorded but not used to tune branch weights or spend another quota.

