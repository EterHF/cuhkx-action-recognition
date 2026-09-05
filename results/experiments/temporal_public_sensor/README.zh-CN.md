# strictV3 Temporal + 公共 Depth/IR

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

本实验保留 strictV3 Temporal 为主干，加入分别微调的公共 Omnivore Swin-T Depth/IR
概率。五个 subject-held 传感器 fold 均从官方公共初始化固定训练 15 epochs，没有加载
历史项目 checkpoint。

完整 OOF accuracy 为 Temporal 0.940053、Depth 0.492754、IR 0.544466。预先固定的
保守概率融合 Temporal/Depth/IR=`0.90/0.05/0.05` 得到 0.939723，少正确 1 条；各 fold
delta 为 A −1、B 0、C +1、D 0、E −1，worst-user 保持 0.8125，未通过逐 fold 非退化
门禁。

Temporal 的 182 个错误中，Depth 修复 33、IR 修复 39、两个传感器共同正确 23；反之，
在 Temporal 正确的样本上，两个传感器共同给出相同错误答案达 428 条。另一次严格 nested
权重诊断仅为 0.939065，并在 outer B/C/D 自动选择传感器权重为零。这确认当前传感器分支
没有可靠的门控信号。

代码保留融合接口，但安全默认值是 sensor gate=0，即完全回退原始 Temporal。本失败
OOF 路线没有构建测试候选，也没有消耗 Kaggle 提交额度。

