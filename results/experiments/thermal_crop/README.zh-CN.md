# 热成像人物裁剪消融

[English](README.md) | [热成像主实验](../thermal_fallback/README.zh-CN.md)

这是一项基于输入几何证据的独立配对消融。按 canonical 顺序每 20 行取一条的训练集
检查发现，现有 YOLO11n 在 142 条有 Thermal 的视频中检出 133 条，覆盖率 93.66%；
包含上下文的裁剪面积从原图的 17.3% 到 100% 不等。该检查没有按标签或错误行选择样本。

唯一变化是在 Thermal 自身坐标中执行人物 union crop，而不是复用 Depth/IR 的窗口。
检测配方直接继承原 Visual 的 8 probe frames、confidence 0.25、margin 1.4、
min-side 0.35；没有检出人物时保留全视野。其余设置与最初 TSN 完全相同，不使用 TSM。
完整训练集检出 2,764/2,891 条有效 Thermal 视频（95.61%），缓存构建全部完成。

三个 seed 的缺失输入接管净纠错为 +3/+9/+3，相对 TSN 的 +10/+8/+13 合计少
16 条正确预测。seed 2026 的按用户聚类 accuracy 增量 95% 区间为
`[0, 0.002009]`。尽管仍没有破坏完整主输入的 logits，这个裁剪配方没有优于
全视野 TSN，且 CI 门槛失败。**不 full-fit、不做匿名推理、不提交 Kaggle**。
没有继续调整检测阈值、margin 或裁剪模式。

完整指标、预测、预注册和权重哈希保留在
[metrics.json](metrics.json)、[predictions.npz](predictions.npz)、
[preregistration.json](preregistration.json)、[provenance.json](provenance.json)。
模型、日志、cache、窗口以及 suite receipt 在 `runs/experiments/thermal_crop_v1/`。
预算不需要额外检测器，复用现有 YOLO 权重；没有构建完整部署 bundle。

## 复现

```bash
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.data.thermal_cache \
  --train-root data/processed/train/HAR/data \
  --metadata results/strict_v3/metadata.npz \
  --output runs/experiments/thermal_crop_v1/inputs --person-crop --device 4
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.thermal_suite \
  --root runs/experiments/thermal_crop_v1 --gpus 4 5 6 7 0 1 2 3
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.evaluation.thermal_fallback \
  --run-root runs/experiments/thermal_crop_v1 --output results/experiments/thermal_crop \
  --control-output results/experiments/thermal_fallback
```

复现需使用新 cache/训练输出目录，避免覆盖或复用此前已完成的训练。
