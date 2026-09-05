# 公共传感器三分支 full-fit

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

使用全部 3,036 条有标签 clip 重新训练三个独立分支：Depth、IR Omnivore Swin-T 仅从
官方公共权重初始化，原生帧率 skeleton 从随机初始化。传感器训练使用随机强度缩放与
偏移增强，skeleton 使用有效帧噪声和分支 dropout；没有加载历史项目 checkpoint。

无增强训练集 accuracy 分别为 Depth 0.931818、IR 0.940711、skeleton 0.833663。
固定等权 logits 的 fp32 ensemble 为 0.950264（2,885/3,036），实际部署 mixed-int8
bundle 为 0.950593（2,886/3,036）。这些都是训练集拟合度，不是泛化估计。

单文件部署 bundle 为 57,925,224 bytes，SHA-256 为 `df994402…`。普通骨干
Conv/Linear 使用按输出通道 int8，敏感传感器层与 skeleton 分支使用 fp16。测试推理
生成 405 条预测，自然覆盖 39/40 类（缺少类别 25）；没有为了补类别而篡改预测。

冻结提交 CSV 的 SHA-256 为 `fe85be36…`。Kaggle submission 56035438 的 public score
为 **0.58706**，与此前约 0.598 的复用 fold 研究诊断处于同一量级，远低于 strictV3
的 0.97512。该候选被否决；排行榜分数只作记录，没有用于调整分支权重或继续消耗额度。

