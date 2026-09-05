# Omnivore Depth/IR 多层融合

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

两个 Omnivore Swin-T trunk 仅从官方公共 checkpoint 初始化。四个中间 stage 的通道数
为 `192/384/768/768`，空间池化后每层保留 8 个时间 token。每层执行 Depth↔IR 双向
四头 cross-attention，四层融合摘要进入统一分类器；最后一层另设两个辅助头保持单模态
监督。

固定 Fold-A 初筛使用 2,320 条训练、716 条 held、batch size 16、seed 2026、15 epochs。训练 accuracy
达到 0.816810，held 仅为 0.512570，worst-user 0.431373；Depth/IR 辅助头 held accuracy
分别为 0.407821、0.502793。融合头确实学到少量跨模态信号，但仍低于独立 IR
0.544693、整段 token fusion 0.539106，以及原 strictV3 Fusion 分支的完整 OOF
0.800395。

失败原因是跨用户泛化，不是未收敛或没有取得多层特征。预先固定的 Fold-A 门禁据此
停止 B–E 扩展、测试推理和 Kaggle 提交。
