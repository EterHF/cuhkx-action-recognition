# 热成像时序位移消融

[English](README.md) | [热成像主实验](../thermal_fallback/README.zh-CN.md)

唯一变化是参考 [TSM](https://arxiv.org/abs/1811.08383)，在 ResNet-18 每个 residual
block 的首个卷积之前，各将 1/8 通道向前、向后移动一帧。clip 边界以零补齐，不跨
clip 混合；模块不增加学习参数。原 TSN 的缓存、采样、增强、15-epoch 配方、三个
seed、用户 folds、缺失输入接管规则和固定部署 seed 均不变。

三个 seed 的接管净纠错分别为 +11/+5/+8，少于 TSN 的 +10/+8/+13；三 seed
合计比对照少 7 条正确预测。固定 seed 2026 的用户聚类 CI 下界仍为零，同时失败于
CI 和相对 TSN 的附加门槛。因此不 full-fit、不运行匿名推理、不提交。

这只否决当前固定配方的 TSM 消融，不能推断时序建模普遍无效；特别是它沿用了
训练四帧、验证八帧的原 TSN 采样设置。没有继续扫描 shift 比例、层位置、采样数或
epoch。逐 seed、fold、user 指标见 [metrics.json](metrics.json)。

复现时建立 `thermal_shift_v1/inputs` 指向控制缓存的符号链接，在 thermal suite
中添加 `--temporal-shift`；评估中添加 `--temporal-shift --control-output
results/experiments/thermal_fallback`。与控制相同的所有设置见
[preregistration.json](preregistration.json)，精确权重和预测哈希见
[provenance.json](provenance.json)。
