# Temporal 条件 Visual 纠错器

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

已实现的约 7.7 万参数纠错头接收 Visual 分类头前特征和冻结的 Temporal logits，向
现有融合 logits 添加 40 类残差。只有最后一层零初始化，因此初始模型严格等于基线，
隐藏投影层仍可正常训练。

训练目标是最终预测的交叉熵，并在训练集中“基线预测正确且置信度不低于 0.8”的样本
上加入 `KL(p0 || p_new)`。评估统计 corrected、broken 与 net，不要求纠错头独立分类。

严格评估现已完成。上游使用完全冻结的外部-only 编码器：Visual 为 PKU-MMD
R(2+1)D-34，骨骼 Temporal 为 NTU DSTFormer；随后在 20 组 outer×inner 划分中重新训练
40 类 head。每条元训练输入都由 inner cross-fit 产生，每个上游 head 都排除 outer-held
用户，checkpoint 固定取终轮；全过程未访问匿名测试输入。

3,036 条 nested-OOF 上，冻结 T+V 基线从 1,201 条正确（`0.395586`）提高到 1,465
（`0.482543`）：纠正 438 条、破坏 174 条，净纠错 +264。A–E 折净变化为
`+46/+58/+48/+66/+46`，18 个用户全部改善。subject-macro 从 `0.391962` 提高到
`0.479934`，worst-user 从 `0.1625` 提高到 `0.29375`；按用户重采样的 accuracy delta
95% 区间为 `[+0.07310,+0.10375]`。第二次同配置运行的两份 OOF 数组逐字节一致。

该结果证明条件残差纠错在严格隔离下有效，但不是 strictV3 部署增益估计。外部-only
代理显著弱于 strictV3，而且代理 Visual 为 1,310 条、Temporal 仅 902 条，冻结的
strictV3 Temporal 主导权重与代理失配。纠错器仍比 Visual 单支多 155 条，但该证据不
授权 full-fit、测试推理或 Kaggle 提交。
