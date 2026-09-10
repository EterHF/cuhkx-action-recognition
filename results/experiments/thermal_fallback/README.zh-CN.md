# 热成像缺失输入补偿，2026-09-10

[English](README.md) | [实验索引](../README.zh-CN.md)

本轮从未用于发布预测的 Thermal 视角寻找新证据。训练集 103 条 Visual/Skeleton
双缺失样本全部包含 Thermal，而 IMU 和 Radar 均缺失。此前只屏蔽缺失输入的训练损失
无法恢复这些样本的动作信息；本轮直接训练独立热成像分类器。

模型为 ImageNet ResNet-18 加时序分段平均分类器。参考
[TSN](https://arxiv.org/abs/1608.00859) 的稀疏采样与视频级监督，使用 16 帧缓存、训练
4 帧、验证 8 帧、160×160 全视野图像；固定 15 epochs、冻结 BN running statistics，
三个 seed 各训练五个互斥用户 fold。每个 fold 从原始 ImageNet 权重独立初始化，
仅最终 FP16 保存状态接受 held-user 评估，没有按 held 分数选 epoch。

推理策略提前固定：只有 Visual 和 Skeleton 同时无有效输入、且 Thermal 有效时，
才采用热成像 top-1；其他行的原 logits 逐元素不变。部署 seed 提前固定为 2026。

| Seed | 双缺失正确数（原为 30/103） | 相对基线净纠错 | 最差用户 accuracy |
| --- | ---: | ---: | ---: |
| 2026 | 40 | +10 | 0.86875 |
| 2027 | 38 | +8 | 0.85000 |
| 2028 | 43 | +13 | 0.86875 |

15 个 seed/fold 组合均未退步，三个 seed 的用户宏平均都改善。但 seed 2026 的
10 条净改善中 9 条来自用户 5；按用户聚类 bootstrap 的 micro-accuracy 增量
95% 区间为 `[0, 0.009521]`，没有通过预注册的严格正下界门槛。
**不晋级、不 full-fit、不运行匿名测试、不提交 Kaggle**。不能在观察结果后改选
CI 为正的 seed 2028。

保留两套明确区分的参照：历史 release-strict 的 2,903/3,036 和当前配对 T+V 的
2,889/3,036。上述净纠错在两套参照上相同，但二者绝对分数不能互换；历史上游
checkpoint 使用过目标标签，组合分数仍不是完全无泄漏的 private-LB 估计。
独立 Thermal 在有效输入上的 held-user accuracy 仅约 37%，本轮没有据此调整
完整输入样本的融合权重。

单个 fold checkpoint 实测 22,444,867 bytes。与原推理包组合在预算上有余量，
但没有构建或声称验证完整部署 bundle。原 0.97512 发布模型及提交文件不变。

配对结构消融另见 [时序位移](../thermal_shift/README.zh-CN.md) 和
[热成像人物裁剪](../thermal_crop/README.zh-CN.md)。本轮不把同一批用户上的多个
消融当作独立的外部确认。

## 复现

在仓库根目录使用 `.conda/envs/cuhkx/bin/python`；缓存和训练输出必须使用新目录。
用户提到的环境在本仓库实际位于 `.conda/envs/cuhkx`。

```bash
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.data.thermal_cache \
  --train-root data/processed/train/HAR/data \
  --metadata results/strict_v3/metadata.npz \
  --output runs/experiments/thermal_fallback_v1/inputs
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.thermal_suite \
  --root runs/experiments/thermal_fallback_v1 --gpus 0 1 2 3 4 5 6 7
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.evaluation.thermal_fallback \
  --run-root runs/experiments/thermal_fallback_v1 \
  --output results/experiments/thermal_fallback
```

原始控制与 TSM 使用相同命令参数的进程队列执行，人物裁剪使用保存 receipt 的 suite
执行。训练日志和权重在 `runs/`；可审查的
[预注册](preregistration.json)、[完整指标](metrics.json)、[逐行预测](predictions.npz)、
[输入与权重哈希](provenance.json) 在本目录。训练全程未使用匿名测试标签或排行榜反馈。
