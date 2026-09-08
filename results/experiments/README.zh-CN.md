# StrictV3 consensus 实验

[English](README.md) | [简体中文](README.zh-CN.md) | [仓库首页](../../README.zh-CN.md) | [技术报告](../../docs/TECHNICAL_REPORT.zh-CN.md)

这些候选来自 Codex 任务 `01a04b41-e2ed-79c3-9fce-20eec40a3c73` 中已审计的工作，并已通过当前公开流水线重放。它们与 `checkpoints/strict_v3/model.pt` 有意分离：`main` 基线仍是可以逐字节复现的 0.97512 发布版本。

| 候选 | OOF | 增益 | Fold 门禁 | 单 checkpoint | 测试变化数 | Kaggle |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| [`sched30_consensus`](sched30_consensus/) | 0.960474 | +0.004282 | 5/5 非退化 | 98,873,941 B | 3 | 0.97014（ref 56016293） |
| [`temporal_pool_consensus`](temporal_pool_consensus/) | 0.959816 | +0.003623 | 5/5 非退化 | 96,936,797 B | 12 | 未提交 |

第一个候选对 strictV3 与固定的三随机种子 sched30 时序发布模型的概率取平均。第二个候选在共享大模型状态的同时，对 motion-energy、top-2-frame 和 top-4-frame 时序池化进行多数投票。

表中的 0.960474 对应冻结并已审计的 artifact。2026-09-03 重新进行的 CPU 训练在校准后
得到 0.959486，原因是数值误差使一个 grid 并列项选择了相邻权重。该结果只作为方向性
证据记录，不替换这里的冻结候选。

审计两个冻结候选：

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.experiments.audit
```

从原始测试数据复现一个候选并验证其冻结 hash：

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.bundle \
  --model checkpoints/experiments/sched30_consensus.pt \
  --output /tmp/inference_bundle.pt
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --bundle /tmp/inference_bundle.pt \
  --output .cache/sched30_consensus.csv
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.experiments.audit \
  sched30_consensus --replayed-csv .cache/sched30_consensus.csv
```

不得根据匿名测试集检查结果晋升或提交任何候选。晋升必须同时满足冻结 OOF 门禁、原始数据精确重放、单 checkpoint 100 MB 包大小门禁，以及明确授权的 Kaggle 提交。

失败的 `highrate_cross_attention` 研究保留为结构证据。恢复并校验后的 strictV3 单分支
OOF 分别为 Fusion 0.800395、Visual 0.896904、Temporal 0.940053。事后去掉 Fusion 的
诊断将冻结融合从2,903提高到2,909行；原生 Skeleton cross-attention 未超过该无 Fusion
基线（2,904行），因此没有进行测试推理。详见
[`highrate_cross_attention/metrics.json`](highrate_cross_attention/metrics.json)。

[`public_backbone_screen`](public_backbone_screen/) 在不读取项目 checkpoint 的前提下
评估公共 Depth/IR 预训练编码器。Omnivore Swin-T 在两个传感器方向均胜出，但固定
fold 结果尚不足以晋升到 `main`。

后续 [`public_sensor_fusion`](public_sensor_fusion/) 保留原生帧率 skeleton 及独立的
Depth/IR Omnivore 分支。整段 token 与时序 token 的 learned cross-attention 均失败；
固定等权 logits 诊断在复用 held fold 上达到 0.597765，但仍属于 selection-biased
证据，必须等待一次全新 full OOF。

经授权的 [`public_sensor_full_fit`](public_sensor_full_fit/) 使用全部 3,036 条数据重训
三支，实际 57.93 MB mixed-int8 bundle 的训练集 accuracy 为 0.950593；Kaggle public
score 仅 0.58706（ref 56035438），因此关闭该路线，不根据排行榜重新调权。

[`temporal_public_sensor`](temporal_public_sensor/) 后续以 strictV3 Temporal 为主干，
固定加入各 5% 的 Depth/IR 概率残差。完整五折 OOF 从 0.940053 降到 0.939723，因此
sensor gate 保持为零，没有构建测试候选。

[`omnivore_layer_fusion`](omnivore_layer_fusion/) 使用双向时序 cross-attention 融合
Depth/IR Omnivore 四个 stage，再进入统一分类器。Fold-A held accuracy 仅 0.512570，
低于独立 IR 和旧 Fusion 分支，因此在 B–E 与测试推理前停止。

[`omnivore_native_rgbd`](omnivore_native_rgbd/) 用重复 IR 加 inverse-depth 启用单个
Omnivore 的原生四通道 token 融合。Fold-A held accuracy 仅 0.379888，表明它与
自然 RGB 加米制 Depth 的预训练契约存在明显域差，因此关闭该路线。

[`temporal_visual_equal`](temporal_visual_equal/) 删除 Fusion，并在冻结 temperature 后
严格等权融合 Temporal/Visual。OOF 降至 0.948946，Kaggle 为 0.95522（ref
56036875），因此不修改 strictV3。

[`temporal_visual_inherited`](temporal_visual_inherited/) 仅将 Fusion 冻结基础权重置零，
继承 strictV3 的 Temporal/Visual temperature、权重与 quality gate。OOF 增加 6 条且
5/5 folds 不退化，Kaggle 持平 0.97512（ref 56036959）。

[`visual_priorities`](visual_priorities/) 按顺序为 Visual R(2+1)D-34 累加 layer2/3/4
时序残差 head、layer4 高时间分辨率以及零初始化 Depth/IR 门控。control 与三个累加
版本的固定 Fold-A accuracy 分别为 0.625698/0.608939/0.618715/0.617318，均未超过
control，因此没有进行测试推理或提交。

[`quantization_budget`](quantization_budget/) 真正删除 Fusion 与未启用 thermal 权重，
得到预测完全一致的 50.73 MB T+V 单 checkpoint。主 Visual 从 5-bit 提至 6/8-bit
没有最终 OOF 收益；可追溯 NTU120 代理则确认 INT4 会损失外部预训练信息，而从原始
浮点重新做 5-bit 可以恢复。

[`conditional_corrector`](conditional_corrector/) 实现了以 Temporal logits 为条件、
零初始化的小型 Visual 残差头。严格 external-only nested OOF 净纠错 264/3,036 条，
所有折和用户均为正向；但代理分支显著弱于 strictV3 且强弱次序不同，因此只验证机制，
不授权部署。后续部署一致筛选中，512 维与 40 维版本分别净下降 24 与 11 条；后者出现
类别 36/用户 5 的集中失败，E 折净下降 32 条，因此未 full-fit、未提交。
