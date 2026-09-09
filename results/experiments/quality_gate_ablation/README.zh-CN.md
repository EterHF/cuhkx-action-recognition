# 固定基础权重的 quality-gate 消融

[English](README.md) | [实验索引](../README.zh-CN.md) | [技术报告](../../../docs/TECHNICAL_REPORT.zh-CN.md)

该机制驱动的探索性消融复用当前 2,889/3,036 配对 T+V OOF，只回答置信度生成的逐样本
quality 因子在既有基础权重之外是否有净贡献。不重训、不拟合参数、不读取匿名测试，也不
扫描权重、温度、阈值、熵、margin 或 top-k 规则。

双主输入有效样本的 control 使用发布 `apply_gate`；candidate 将两支 quality 因子设为
相同常数。Temporal/Visual 基础权重保持 0.67/0.22，温度保持
1.1651778/1.5271352，Visual 发布输出缩放保持 0.5。103 条双主输入缺失和 2 条部分缺失
样本保留逐元素一致的 control logits。

## 结果

| 指标 | Quality gate 开启 | Quality gate 关闭 | 变化 |
| --- | ---: | ---: | ---: |
| 正确数 | 2,889 / 3,036 | 2,893 / 3,036 | +4 |
| 用户宏平均准确率 | 0.950617 | 0.952000 | +0.001384 |
| 最差用户准确率 | 0.8125 | 0.8125 | 0 |

候选纠正 6 条、破坏 2 条，各折净变化为 `+1/0/+1/+1/+1`。原 53 条 Visual 独有
top-1 正确错误中修复 5 条，6 条 Temporal 独有正确中修复 1 条；15 条两支 top-1 都错和
73 条双主输入缺失错误均未修复。原先 2,859 条双主输入有效且预测正确的样本中破坏 2 条。

数学一致性检查通过：8 个变化全部位于分支分歧样本；两支具有相同唯一 top-1 的 2,655
条样本变化为 0，双主输入有效子集之外变化也为 0。取消 gate 后 Visual 有效权重固定为
0.24719；control 的逐样本 Visual 有效权重范围为 0.08435–0.32395。

该小幅正结果不足以晋级：双侧精确 McNemar p=0.2891，按用户聚类 bootstrap 的 95%
区间为 `[-0.000675, 0.003083]`。两条破坏均来自用户 18，使该用户净 -2；尽管四折各净增
1、一折持平，预注册 bootstrap 门槛仍失败。因此不构建发布候选、不运行测试推理、不
提交，也不继续扫描复杂融合规则。

证据见 [preregistration.json](preregistration.json)、[metrics.json](metrics.json) 和
[changes.csv](changes.csv)。可复现入口为
[`quality_gate_ablation.py`](../../../yolo_r2plus1d/strict_v3/evaluation/quality_gate_ablation.py)。
