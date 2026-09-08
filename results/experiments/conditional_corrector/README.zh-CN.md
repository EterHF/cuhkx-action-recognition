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
固定种子 2026/2027/2028 的净纠错分别为 +264/+261/+276，15 个折×种子组合全部为正。

该结果证明条件残差纠错在严格隔离下有效，但不是 strictV3 部署增益估计。外部-only
代理显著弱于 strictV3，而且代理 Visual 为 1,310 条、Temporal 仅 902 条，冻结的
strictV3 Temporal 主导权重与代理失配。纠错器仍比 Visual 单支多 155 条，但该证据不
授权 full-fit、测试推理或 Kaggle 提交。

另一次部署一致筛选从原始各折 checkpoint 重新构建 uniform-5-bit Visual OOF logits，
并严格使用发布版单视图预处理。冻结 T+V 基线为 2,890/3,036（`0.951910`），且测试集
预测与此前 T+V 提交逐行一致。512 维纠错器净下降 24 条；唯一预注册的坐标对齐改动
（改用 40 维 Visual 类别 logits）仍净下降 11 条（纠正 32、破坏 43），A–E 折净变化为
`+6/+9/+6/0/-32`。subject-macro 从 `0.951297` 降至 `0.947341`，worst-user 从
`0.8125` 降至 `0.625`。

该失败具有集中偏置，并非可忽略的随机波动：用户 5 中 30 条原本预测正确的类别 36
被纠错器改成类别 10，主导了 E 折退化，直接违背未知用户泛化目标。因此两个预注册的
部署版本均未晋级，未执行 full-fit，也未消耗 Kaggle 提交次数。精确机器可读结果见
[`deployment_screen.json`](deployment_screen.json)。
