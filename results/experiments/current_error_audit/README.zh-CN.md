# 当前 T+V 错误与输入证据审计

[English](README.md) | [实验索引](../README.zh-CN.md) | [技术报告](../../../docs/TECHNICAL_REPORT.zh-CN.md)

本审计严格采用当前配对的 2,889/3,036 T+V OOF 基线，不沿用旧发布模型 2,903 条正确
预测对应的错误名单；不重训、不读取匿名测试、不调整融合，也不拼接历史候选预测。

147 个错误中，73 个位于 103 条双主输入缺失样本。其余 74 个里，只有 Temporal top-1
正确 6 条，只有 Visual top-1 正确 53 条，两支 top-1 都错 15 条。这 15 条的真实类别在
Temporal 中全部排名第 2；Visual 中 11 条排名第 2、1 条第 3，另三条分别为第
13/25/36。因此，有效输入剩余错误主要是已有分支证据未被最终组合保留，而不是两支都把
真实类别排得很后。该覆盖统计只是描述，既不是软融合准确率上限，也不授权普遍提高
Visual 权重。

六个可比的近期候选合计只曾纠正 10 个基线错误，137 个始终未纠正。持续错误包括全部
73 个双主输入缺失错误、6 个 Temporal 独有正确、45 个 Visual 独有正确和 13 个两支
top-1 都错。逐行记录见 [errors.csv](errors.csv)；不得将其作为 oracle ensemble 或纠错器
训练名单。

输入证据盲审从 10 个独立的持续双错混淆簇各抽取一条，并为每条匹配同类、不同用户的
正确对照。解盲前并排检查全部 Depth_Color 原始帧、实际选择的 16 张未裁剪帧、union
crop 后 128×128 Depth/IR cache 和骨架。没有发现明确的采样或空间处理损失；骨架可疑
异常在错误和对照中均为 2/10，20 条样本的各模态源帧数全部一致。该小规模描述性审计
不授权启动采样、局部视图、分辨率或骨架修复训练。

证据见 [metrics.json](metrics.json)、[review_summary.json](review_summary.json)、
[blind_observations.csv](blind_observations.csv) 和
[review_answer_key.csv](review_answer_key.csv)。

可复现分析入口为
[`current_error_audit.py`](../../../yolo_r2plus1d/strict_v3/evaluation/current_error_audit.py)
和
[`build_evidence_review.py`](../../../yolo_r2plus1d/strict_v3/evaluation/build_evidence_review.py)。
生成的 contact sheets 保留在 `runs/` 而不提交；审阅摘要、盲态观察及解盲表保留在本目录。
