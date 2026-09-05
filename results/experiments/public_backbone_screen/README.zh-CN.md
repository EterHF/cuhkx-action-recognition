# 公共 Depth/IR 主干初筛

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

本实验保持骨骼路径不变，筛选可替换旧 Visual/Fusion 计算的公共预训练编码器。
实验不读取任何项目历史 checkpoint，不查看匿名测试集，也不用 held fold 标签选择
epoch。

固定 users 1、6、17、22 为单个跨主体验证 fold。冻结初筛时，每个编码器都配同一个
随机初始化的时序 probe，固定训练 25 epochs；然后只对 Depth 和 IR 各自胜出的模型
固定微调 15 epochs。

| 公共编码器 | 传感器 | 参数量 | 冻结准确率 | 最差用户 |
| --- | --- | ---: | ---: | ---: |
| DeFM EfficientNet-B0 | Depth | 3.01 M | 0.25000 | 0.20197 |
| DeFM RegNetY-800MF | Depth | 6.25 M | 0.25279 | 0.21675 |
| DeFM ResNet-18 | Depth | 11.74 M | 0.25000 | 0.17734 |
| DeFM ViT-S/14 | Depth | 21.64 M | 0.25279 | 0.21182 |
| DFormerv2-S | Depth | 25.42 M | 0.28212 | 0.24631 |
| DFormerv2-B | Depth | 52.60 M | 0.26676 | 0.21675 |
| DFormerv2-L | Depth | 93.78 M | 0.21508 | 0.17734 |
| **Omnivore Swin-T** | **Depth** | **27.85 M** | **0.34777** | **0.25616** |
| **Omnivore Swin-T** | **IR** | **27.85 M** | **0.43296** | **0.33333** |
| M-SpecGene ViT-B | IR | 85.80 M | 0.24441 | 0.18301 |

胜者微调后，Depth Omnivore 达到 **0.47207**，IR Omnivore 达到
**0.54469**。固定 25% Depth / 75% IR 的概率晚期融合达到 **0.55307**，
最差用户准确率为 **0.43791**。所有朴素特征拼接均低于冻结的 IR 单支，因此否决
拼接式 Fusion。

这是模型初筛，而不是无偏的泛化估计：同一个 held fold 用于模型家族排序，25%/75%
诊断权重也来自该 fold 上的冻结 probe 探索网格。epoch 数和最终 checkpoint 虽然固定，
但晋升前仍需要全新未使用切分或完整 OOF。

DFormerv2 的 Small、Base、Large 三种规模均已测试。Base 和 Large 都低于 Small，
因此不能用参数量代替迁移效果验证。

由此保留的研究架构方向是：高采样率骨骼分支、独立微调的 Omnivore Depth 分支、
独立微调的 Omnivore IR 分支。Cross-attention 是下一阶段实验，不能由本次初筛直接
宣称有效。当前候选不会晋升到 `main`，因为单 fold 结果存在选择偏差，且分数尚不足以
替换 strictV3。

原始特征、logits 和 checkpoint 位于被忽略的
`runs/experiments/public_backbone_screen/`；提交到仓库的
[`metrics.json`](metrics.json) 记录了公共权重 SHA-256 和全部报告分数。安装可选依赖：

```zsh
python -m pip install -e '.[public-backbones]'
```

查看完整命令：

```zsh
python -m yolo_r2plus1d.strict_v3.training.public_backbone_screen --help
python -m yolo_r2plus1d.strict_v3.training.finetune_public_omnivore --help
```

InfMAE 的官方预训练 checkpoint 仅通过百度网盘发布，本环境未能取得；Thermal-MAE
权重受限并返回 HTTP 403；未核实到稳定的 DuGI-MAE 官方公开权重。ViT-Lens Depth
采用 ViT-L，在保留两路传感器模型、骨骼分支和检测器的 100 MB 总预算下性价比不足。
M-SpecGene 的 1.44 GB 发布文件包含优化器和双解码器，本实验只加载并验证其中
85.80 M 参数的编码器。

两个 Omnivore 分支的 int8 参数载荷估算为 55.77 MB；加上保留的骨骼、时序和检测器
后约为 66.42 MB（尚未计序列化开销）。预定策略是骨干 Conv/Linear 按输出通道 int8，
输入 patch、相对位置偏置、分类头、归一化、偏置、融合和骨骼分支保留 fp16；只有实际
序列化包仍超限时才退到 int6。
这只通过了参数算术初筛，尚未构建或认证符合官方规则的提交包。

公共来源：[DeFM](https://github.com/leggedrobotics/defm)、
[DFormer](https://github.com/VCIP-RGBD/DFormer)、
[Omnivore](https://github.com/facebookresearch/omnivore) 和
[M-SpecGene](https://github.com/CalayZhou/M-SpecGene)。公共权重只下载到 `.cache/`，
本仓库不再分发。Omnivore 使用 CC BY-NC 4.0，因此任何衍生 checkpoint 都必须保留
署名与非商业限制；这是正式发布门禁，而不只是文档备注。
