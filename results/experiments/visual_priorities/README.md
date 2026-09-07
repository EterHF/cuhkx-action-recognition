# Ordered strictV3 Visual architecture priorities

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

This study tests three cumulative Visual changes in their declared order. All
arms start from the same public IG-65M to Kinetics-400 R(2+1)D-34 encoder; no
project checkpoint is loaded. The 2,320 training rows and 716 held rows are
split by subject, preprocessing statistics are training-only, and the held
labels are evaluated once after a fixed 15 epochs rather than used to select a
checkpoint.

| Arm | Cumulative change | Train accuracy | Held correct | Held accuracy | Worst user |
| --- | --- | ---: | ---: | ---: | ---: |
| control | original global-pool head | 0.959914 | 448/716 | **0.625698** | **0.517241** |
| multiscale | layer2/3/4 temporal residual head | 0.959483 | 436/716 | 0.608939 | 0.497537 |
| highres | + remove only layer4 temporal stride | 0.957328 | 443/716 | 0.618715 | 0.497537 |
| modality_gate | + bounded zero-initialized Depth/IR gate | 0.958621 | 442/716 | 0.617318 | 0.497537 |

The multiscale head spatially pools layer2/3/4, projects each stage to 128
channels, resamples them to the eight-token layer2 timeline, then applies two
depthwise temporal convolutions and attention pooling. Its classifier is zero
initialized, so it begins as an exact residual identity. The high-resolution
arm changes layer4's temporal strides from two to one while retaining its
spatial stride. The final 114-parameter gate uses clip-level Depth/IR mean and
standard deviation and bounds each sensor scale to `[0.75, 1.25]`; its output
is also exactly identity at initialization.

High-resolution layer4 recovered 7 rows relative to the failed multiscale arm,
which is a useful directional signal, but remained 5 rows below the unmodified
control. The modality gate lost one more row. None passed Fold A, so strictV3
is unchanged and no B-E training, full fit, anonymous-test inference, or
Kaggle submission was performed.

Run one arm with:

```zsh
.conda/envs/cuhkx/bin/python -m \
  yolo_r2plus1d.strict_v3.training.visual_priorities \
  --variant control \
  --cache /path/to/train_depth_ir.npy \
  --source-encoder /path/to/public_encoder.pt \
  --preprocessing-contract /path/to/contract_foldA.json \
  --output-dir runs/experiments/visual_priorities/control
```
