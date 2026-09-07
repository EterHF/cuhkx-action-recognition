# strictV3 Visual 架构优先级顺序实验

[English](README.md) | [简体中文](README.zh-CN.md) | [实验索引](../README.zh-CN.md)

本研究按声明顺序累加验证三项 Visual 改动。所有实验均从同一份公开的
IG-65M → Kinetics-400 R(2+1)D-34 编码器开始，不读取项目 checkpoint。按 subject
划分 2,320 条训练数据和 716 条 held 数据，预处理统计量只由训练部分拟合；固定训练
15 轮后仅评估一次 held labels，不用它选择 checkpoint。

| 实验 | 累加改动 | 训练 accuracy | Held 正确数 | Held accuracy | 最差用户 |
| --- | --- | ---: | ---: | ---: | ---: |
| control | 原始全局池化 head | 0.959914 | 448/716 | **0.625698** | **0.517241** |
| multiscale | layer2/3/4 时序残差 head | 0.959483 | 436/716 | 0.608939 | 0.497537 |
| highres | + 仅移除 layer4 时间 stride | 0.957328 | 443/716 | 0.618715 | 0.497537 |
| modality_gate | + 有界、零初始化 Depth/IR 门控 | 0.958621 | 442/716 | 0.617318 | 0.497537 |

多尺度 head 对 layer2/3/4 做空间池化，各自投影到 128 维并重采样到 layer2 的 8 个
时间 token，再使用两个 depthwise 时序卷积和 attention pooling。分类器为零初始化，
因此初始时是严格恒等的残差。高分辨率版本只把 layer4 的时间 stride 从 2 改为 1，
保留空间 stride。最后的 114 参数门控读取每段 clip 的 Depth/IR 均值与标准差，将各
模态缩放限制在 `[0.75, 1.25]`，初始输出同样严格等于输入。

高时间分辨率 layer4 相对失败的 multiscale 版本找回 7 条，说明方向有一定信号，但仍
比未修改 control 少 5 条；模态门控又少 1 条。三项均未通过 Fold A 门禁，因此不修改
strictV3，也没有进行 B–E、full-fit、匿名测试推理或 Kaggle 提交。

运行单个实验：

```zsh
.conda/envs/cuhkx/bin/python -m \
  yolo_r2plus1d.strict_v3.training.visual_priorities \
  --variant control \
  --cache /path/to/train_depth_ir.npy \
  --source-encoder /path/to/public_encoder.pt \
  --preprocessing-contract /path/to/contract_foldA.json \
  --output-dir runs/experiments/visual_priorities/control
```
