# 仅有效主输入的纠错器

[English](README.md) | [简体中文](README.zh-CN.md)

旧 40 维 semantic corrector 的 43 条破坏中，30 条来自用户 5 的双主输入缺失样本。
本实验沿用 60 epochs、128 hidden、原 CE+KL，统一限定训练和推理都需要两个主模态有效。
没有按类别、用户或匿名 ID 路由；缺失行的 baseline logits 完全不变。

3 seeds × 5 folds 共 15 个模型相对部署对齐基线 2,890/3,036，净纠错 +17/+20/+17。
E 折仍为 -3/-3/-1；固定部署 seed 的用户聚类 95% CI 为 [-0.0661,+1.2287] 个百分点。
因此拒绝单独部署，未 full-fit、未匿名推理、未提交。进一步失误诊断用于明确提出后续
二选一选择器，不能把这些反复使用的 held 数据视为新确认性数据。

Evidence: [preregistration.json](preregistration.json), [metrics.json](metrics.json), [predictions.npz](predictions.npz).
