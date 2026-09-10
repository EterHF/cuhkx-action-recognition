# 受约束的双分支选择器

[English](README.md) | [简体中文](README.zh-CN.md)

仅在两个主模态都有效、两个模型 top1 不同、baseline 置信度低于 0.8 时启用六维输入的
6→16→2 MLP；选择范围严格限定为 Temporal 或 Visual 的 top1。0.8 沿用原纠错器保护阈值，
没有扫描。训练只使用真值属于两个候选之一的分歧行，固定 100 epochs、AdamW .01。

原 3 seeds × 5 folds 的净纠错为 +23/+25/+27；用户聚类 95% CI 均高于 0。但后两个 seed
的 D 折均 -1，因此单模型方案未通过全部门槛，没有单独 full-fit 或提交。
后续另立 `selector_consensus` 实验，采用全部三个固定 seed 一致才改判；新增 2029–2031
三种子复制检验用于检查初始化稳定性，未按结果选择最佳 seed。

Evidence: [preregistration.json](preregistration.json), [metrics.json](metrics.json), [predictions.npz](predictions.npz).
