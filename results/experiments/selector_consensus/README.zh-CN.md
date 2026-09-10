# 三种子一致的受约束选择器

[English](README.md) | [简体中文](README.zh-CN.md)

三个固定选择器只有全部同意同一个不同于 baseline 的答案时才改判，否则精确保留 baseline。
部署种子为 2026/2027/2028；在锁定一致规则后另训 2029/2030/2031 作初始化稳定性复制检验。

两个 triplet 都相对部署对齐基线 2,890/3,036 净增加 26 条，达到 2,916/3,036（96.0474%）。
A–E 净增都为 +6/+13/+6/0/+1，subject macro 改善、worst-user 不退化；用户聚类 95% 区间
为 +0.3560 至 +1.4459 个百分点。部署 triplet 纠正 35、破坏 9；复制 triplet 纠正 37、破坏 11。

完整训练复用了固定配方，每个 head 使用 264 条可辨别分支优劣的训练分歧样本。原始训练
有效性重建与 OOF mask 逐项一致。单文件打包全部权重（包括 YOLO）为 50,730,395 B。
两次独立原始重放与 Kaggle 验证状态以 `deployment.json` 为准。

这是在历史错误诊断后提出的探索性机制。六组种子检查初始化稳定性，**不是独立数据集**；
上游模型仍有目标预训练污染，因此不能将 OOF 提升或区间直接解释为 private LB 保证。
原 canonical 发布文件与 0.97512 记录保持可复现。部署代码入口为
`python -m yolo_r2plus1d.strict_v3.release.replay --bundle <candidate_bundle> --output <csv>`。

Evidence: [preregistration.json](preregistration.json), [metrics.json](metrics.json), [predictions.npz](predictions.npz).

公开验证：2026-09-10 提交 **56145116**，分数 **0.96019**，低于原最佳 **0.97512**。拒绝晋级，冻结该方向，不根据公开反馈调参或重提。当前最佳 checkpoint/CSV 未覆盖。private score 尚不可用。
