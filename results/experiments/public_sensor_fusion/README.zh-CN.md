# 独立公共传感器融合

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

本研究保留原生帧率 skeleton 分支，并分别微调独立的 Depth 与 IR Omnivore Swin-T。
三个目标模型均从公共初始化或随机初始化开始训练，没有加载历史项目 checkpoint。固定
held users 为 1、6、17、22（716 行），固定 epoch 30、seed 2026；未访问匿名测试集。

| 候选 | Accuracy | Worst user | 结论 |
| --- | ---: | ---: | --- |
| 高频 skeleton-only | 0.455307 | 0.359477 | 有互补性，不能单独使用 |
| Depth Omnivore | 0.472067 | 0.372549 | 保留为独立多样性分支 |
| IR Omnivore | 0.544693 | 0.418301 | 最强独立传感器分支 |
| 两个整段 token + cross-attention | 0.539106 | 0.411765 | 否决 |
| 8+8 时序 token + cross-attention | 0.515363 | 0.392157 | 过拟合，否决 |
| Depth + IR + skeleton 等权 logits | **0.597765** | **0.503268** | 仅研究候选 |

时序版为每个传感器保留 8 个 Swin 后端时间 token，并保留最多 256 个有序 skeleton
帧；但训练准确率达到 0.9651 时，跨用户泛化反而低于整段 token 版本。因此下一版应将
独立监督的分支 logits 作为主路径，cross-attention 只能作为可关闭的残差，不能替代
独立分类器。

等权 logits 诊断运行前，本 held fold 已被查看，因此结果明确标为 selection-biased。
它只能证明错误具有互补性，不能当作可晋升 OOF，也不能据此提交 Kaggle。

混合 int8 将骨干 Conv/Linear 按输出通道量化为 int8，敏感的输入 patch、相对位置
偏置、分类头、norm 和 bias 保持 fp16。序列化后的 Depth、IR 分支分别为 28,504,203
与 28,503,495 bytes，合计 57,007,698 bytes；加上保留的 skeleton/temporal/detector
估算为 67,657,705 bytes（尚未计最终 bundle 开销）。这只是组件级大小初筛，不代表
官方单 checkpoint 已认证。
在相同 batch-16 推理下，混合 int8 使 Depth 从 0.472067 变为 0.467877，IR 的整体
accuracy 保持 0.544693。

复现入口：

```zsh
python -m yolo_r2plus1d.strict_v3.training.public_sensor_fusion --help
python -m yolo_r2plus1d.strict_v3.evaluation.public_sensor_ensemble --help
```
