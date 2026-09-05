# strictV3 Temporal + Visual 等权融合

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

本消融删除 legacy Fusion 分支，在冻结的逐 fold/full temperature 后，对现有 strictV3
Temporal 和 Visual logits 固定使用 `0.50/0.50`。关闭逐样本 quality gate，因此部署
权重严格等同。没有重新训练模型；复用的 strictV3 bundle 为 69,805,793 bytes，低于
官方 100 MB 限制。

五折 subject OOF 为 2,881/3,036（`0.948946`），低于 release strictV3 的
2,903/3,036（`0.956192`）。A–E fold accuracy 为 `0.959497/0.936107/0.979381/`
`0.955010/0.901879`。测试预测自然覆盖全部 40 类，相对 strictV3 改变 15/405 条。

按用户明确要求提交 Kaggle，ref `56036875` 得分为 `0.95522`，低于 strictV3 的
`0.97512`。该消融已否决，且不根据此分数继续调整下一组融合权重。
