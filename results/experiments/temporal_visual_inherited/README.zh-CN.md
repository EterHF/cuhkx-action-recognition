# strictV3 Temporal + Visual 继承式门控优化

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

该候选删除较弱的 legacy Fusion，不拟合任何新参数。strictV3 的逐 fold/full
temperature、Temporal/Visual 基础权重与置信度 quality gate 全部冻结；唯一变化是将
Fusion 基础权重置零，由原 gate 归一化剩余两支。

Subject-wise OOF 从 2,903/3,036（`0.956192`）提升至 2,909/3,036
（`0.958169`）。A–E fold delta 为 `0/0/+5/+1/0`；macro recall 从 0.953136
升至 0.956076，subject-macro 从 0.955794 升至 0.957954，worst-user 保持
0.8125。它优于已否决的严格等权方案，因为 Temporal 仍占主导：经过置信度门控后，
测试集 Visual 平均有效权重只有 0.1653。

复用未改变的 69,805,793-byte strictV3 bundle。测试预测覆盖全部 40 类，仅相对
strictV3 改变 3/405 条。通过逐 fold 门禁后提交 Kaggle，ref `56036959` 得分
`0.97512`，与 strictV3 持平。该方案作为更简洁的 private-LB 候选保留，不根据公开榜
持平结果继续搜索权重。
