# 删除 Fusion 后的部署与量化预算

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

当前已提供并验证真正删除 `fusion_4bit` 与未启用 thermal 占位权重的发布构建路径。
保留的 Temporal/Visual temperature、基础权重、quality gate 以及
`visual_package_output_scale=0.5` 均未改变。得到的模型为 45,111,978 bytes；嵌入
YOLO 后单 checkpoint 为 50,730,599 bytes，余量 49,269,401 bytes。405 条预测与此前
仅将 Fusion 权重置零的已提交候选完全一致。

随后所有量化均从原始 FP16 权重重新进行，绝不从低比特包反量化。3,036 条
subject-wise OOF 结果为：

| 方案 | Visual 正确数 | T+V 正确数 | Bundle bytes |
| --- | ---: | ---: | ---: |
| FP16 参考 | 2,754 | 2,893 | 超预算/未构建 |
| uniform 5-bit | **2,755** | **2,893** | 50,732,895 |
| uniform 6-bit | 2,747 | 2,892 | 58,666,015 |
| uniform 8-bit | 2,754 | 2,893 | 74,532,191 |
| 5-bit + layer4/head 8-bit | 2,754 | 2,893 | 65,385,317 |
| 6-bit + layer4/head 8-bit | 2,746 | 2,892 | 68,437,719 |

更高精度虽然都能装下，但没有产生正的最终净纠错，因此主 Visual 保持 5-bit。

附件中 FP16 1,998、INT4 1,859 的精确候选在工作区缺少原始浮点 checkpoint，不能
合法重新量化。明确标记的可追溯 NTU120 代理仍验证了机制：FP16 为 1,788，INT4 为
1,730，从浮点重新做 uniform 5-bit 达到 1,804；仅将 layer4/head 提至 8-bit 反而只有
1,716，说明 INT4 损失并不只集中在末层。由于简单 5-bit 已消除代理上的 FP16 accuracy
缺口，暂不增加 BRECQ 激活重建；附件模型仍需其原始 FP16/FP32 checkpoint。
