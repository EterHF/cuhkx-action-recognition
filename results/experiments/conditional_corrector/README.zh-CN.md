# Temporal 条件 Visual 纠错器

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

已实现的约 7.7 万参数纠错头接收 Visual 分类头前特征和冻结的 Temporal logits，向
现有融合 logits 添加 40 类残差。只有最后一层零初始化，因此初始模型严格等于基线，
隐藏投影层仍可正常训练。

训练目标是最终预测的交叉熵，并在训练集中“基线预测正确且置信度不低于 0.8”的样本
上加入 `KL(p0 || p_new)`。评估统计 corrected、broken 与 net，不要求纠错头独立分类。

目前有意没有开始训练。合法的五折评估需要 20 组 outer×inner 上游运行，使纠错器的
outer-train 输入在折内交叉生成，同时每个上游模型也排除相应 outer-held 用户。当前
缺少这些特征/logits；训练入口会拒绝普通全局 OOF manifest，而不是静默产生有泄漏
偏差的 stacking 结果。
