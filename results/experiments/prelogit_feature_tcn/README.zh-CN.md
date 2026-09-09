# 分类前特征 TCN

[English](README.md) | [实验索引](../README.zh-CN.md) | [技术报告](../../../docs/TECHNICAL_REPORT.zh-CN.md)

该五折配对 OOF 实验只检验一个结构问题：发布 DSTFormer 在最终 40 类单帧分类器之前，
是否保留了 frame logits 无法供 TCN 利用的时序证据。

对照为 `mean(frame_logits) + TCN(frame_logits)`；候选保留完全相同的静态均值，唯一变化是
TCN 改读 `PCA40(pre_fc2_features)`。每个 PCA 只在外层训练用户的帧上拟合，随后冻结。
两种输入由同一次 DSTFormer 前向导出；对照 logits 与既有缓存最大差异为 0，SHA256 也
完全一致。TCN 结构与参数量、16 帧采样、随机种子、五轮预算、Visual OOF 和发布 T+V
融合契约全部固定。

## 结果

| 指标 | Logit TCN | Feature TCN | 变化 |
| --- | ---: | ---: | ---: |
| Temporal 正确数 | 2,847 / 3,036 | 2,839 / 3,036 | -8 |
| 冻结 T+V 正确数 | 2,889 / 3,036 | 2,891 / 3,036 | +2 |
| T+V 用户宏平均准确率 | 0.950617 | 0.951393 | +0.000777 |
| T+V 双主输入有效正确数 | 2,859 / 2,931 | 2,861 / 2,931 | +2 |
| T+V 双主输入缺失正确数 | 30 / 103 | 30 / 103 | 0 |

最终 T+V 纠正 6 条、破坏 4 条；各折净变化为 `A 0、B -1、C +2、D 0、E +1`，因此
形式上通过预注册门槛。但这不是稳健增益：总共仅改变 10 个预测，双侧精确 McNemar
检验 p=0.7539，按用户聚类 bootstrap 的准确率变化 95% 区间为
`[-0.000983, 0.002628]`。

机制证据是混合的，而非得到确认。候选 Temporal 单支在三折下降、两折持平；最终的微小
增益只在与冻结 Visual 交互后出现。变化分布在六个用户，但用户 8 的两条“洗脸”纠正被
同一用户两条“擦手”破坏抵消，“阅读文档／翻页”也发生双向变化。因此不晋级、不做
full-fit、不提交，也不继续扫描 PCA 维数或 attention 变体；“40 维 logits 是主要时序
瓶颈”的优先级应当降低。

机器可读证据见 [metrics.json](metrics.json)、[stability.json](stability.json)、
[preregistration.json](preregistration.json)、[feature_export.json](feature_export.json) 和
[input_subset_decomposition.json](input_subset_decomposition.json)。

## 复现实验输入

在仓库根目录使用 zsh 执行。输出目录必须不存在，从而避免静默复用旧 PCA cache。

```zsh
.conda/envs/cuhkx/bin/python -m \
  yolo_r2plus1d.strict_v3.data.prelogit_features \
  --package checkpoints/strict_v3/model.pt \
  --output-dir runs/experiments/prelogit_feature_tcn_v1/inputs \
  --reference-logits \
    runs/experiments/skeleton_retargeting_v2/inputs/original_frame_logits.npy \
  --device cuda:0
```

随后用 `yolo_r2plus1d.strict_v3.training.temporal` 将每折训练两次：control 只传入
`--logits .../frame_logits.npy`；candidate 额外传入本折的
`--temporal-input .../foldX_pca40.npy`。二者均采用 `--kind tcn --epochs 5
--scheduler-epochs 30 --seed 2026 --select-last --defer-val-metrics`，held-user
划分见 [preregistration.json](preregistration.json)。
