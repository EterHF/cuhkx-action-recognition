# Omnivore 原生 RGB-D token 融合

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

单个官方 Omnivore Swin-T trunk 接收对齐的 `[IR, IR, IR, inverse-depth]` clip。
四通道输入会启用 Omnivore 原生 `summed_rgb_d_tokens` 路径：appearance 与 Depth
分别完成 patch embedding，然后在共享 transformer 前相加。归一化统计仅由 Fold-A
的 2,320 条训练样本拟合，没有加载项目历史 checkpoint。

固定 15 epochs、batch size 16 的增强训练 accuracy 为 0.662069。在 716 条跨用户
held 样本上，accuracy 为 0.379888，worst-user 为 0.290640，低于独立 IR
（0.544693）、双 trunk 多层 Fusion（0.512570）和 strictV3 Visual（完整 OOF
0.896904）。

原生路径在机制上运行正确，但其预训练契约是假设自然 RGB 加米制 Depth；用重复 IR
替代 RGB、用 JET 反解的伪深度替代米制 Depth，会产生早期融合域冲突。Fold-A 门禁据此
停止 B–E 训练、匿名测试推理和 Kaggle 提交。
