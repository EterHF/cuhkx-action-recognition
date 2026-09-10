# 本地／世界坐标 IMU 配对实验

[English](README.md) | [简体中文](README.zh-CN.md)

对 2,792 条至少有三个有效传感器的训练片段，使用六个 InceptionTime 风格模块，配对比较
重复本地坐标与本地+四元数世界坐标。每个分支 126,632 参数、273,637 B，固定 80 epochs，
3 seeds × 5 folds × 2 views，共 30 个模型。每个外层模型排除对应用户。

四元数方向通过 13,053 条读数的重力检验，未使用绝对时间、硬件身份、温度、电量等元数据。
世界坐标的平均单支 accuracy 为 30.8739%，低于本地的 31.4112%。固定 10% 概率融合相对
历史 canonical 的净纠错为 +2/+1/+1，相对当前配对 T+V 为 +1/+2/0，未通过锁定门槛。
因此未 full-fit、未测试推理、未提交，也未扫描融合权重。

模型/cache/log 位于 `runs/experiments/imu_global_v1/`。历史基线与当前配对基线分别报告，
没有将两者绝对准确率混用。主模型上游已见过目标标签，这些融合 OOF 仅是工程证据。

参考：[InceptionTime](https://arxiv.org/abs/1909.04939)、[FLOW](https://arxiv.org/abs/2406.18569)。

Evidence: [preregistration.json](preregistration.json), [metrics.json](metrics.json), [predictions.npz](predictions.npz).
