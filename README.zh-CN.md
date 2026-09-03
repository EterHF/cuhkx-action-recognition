# CUHK-X 小模型赛道 — strictV3

[English](README.md) | [简体中文](README.zh-CN.md) | [文档索引](docs/README.zh-CN.md)

这是 CUHK-X 小模型赛道（UbiComp / ISWC 2026）清理后的 strictV3 基线。发布包可以从原始测试数据逐字节复现正式提交 `55712568`，其公开榜分数为 **0.97512**、排名第 3。可部署包连同 YOLO11n 检测器共占 69.82 MB。

主代码树有意排除了历史实验和已否决方向；它们的方法与结果保存在[技术报告](docs/TECHNICAL_REPORT.zh-CN.md)（[英文版](docs/TECHNICAL_REPORT.md)）中。

## 仓库结构

```text
checkpoints/strict_v3/       发布模型与检测器（Git LFS）
results/strict_v3/           OOF/test logits、指标及标准 CSV
yolo_r2plus1d/strict_v3/
├── data/                    确定性索引与 cache 构建器
├── models/                  R(2+1)D、DSTFormer、融合与优化器代码
├── training/                视觉、Skeleton、时序与融合训练
├── inference/               分支级推理
├── release/                 原始数据重放、融合契约与验证
└── cli/                     带保护措施的 Kaggle 提交辅助工具
tests/                       快速发布与数据契约测试
docs/TECHNICAL_REPORT.md     实验历史与设计依据
```

## Git 点击导航

| 目标 | 链接 |
| --- | --- |
| strictV3 源码 | [`yolo_r2plus1d/strict_v3/`](yolo_r2plus1d/strict_v3/) |
| 训练入口 | [`training/`](yolo_r2plus1d/strict_v3/training/) |
| 发布与验证 | [`release/`](yolo_r2plus1d/strict_v3/release/) |
| 测试 | [`tests/`](tests/) |
| 发布 checkpoint | [`checkpoints/strict_v3/`](checkpoints/strict_v3/) |
| 结果与 manifest | [`results/strict_v3/`](results/strict_v3/) |
| 数据布局 | [`data/README.zh-CN.md`](data/README.zh-CN.md) |
| 重训练指南 | [`docs/RETRAINING.zh-CN.md`](docs/RETRAINING.zh-CN.md) |
| 技术报告 | [`docs/TECHNICAL_REPORT.zh-CN.md`](docs/TECHNICAL_REPORT.zh-CN.md) |

生成的 cache、数据集和训练运行不会纳入版本控制。

分支 `experiment/strictv3-consensus` 维护两个通过 OOF 资格检查的 consensus 候选。其冻结包、原始重放 hash 和审计命令记录在 [`results/experiments/README.zh-CN.md`](results/experiments/README.zh-CN.md)；两者均未提交 Kaggle。

## 安装

```bash
git lfs install
git lfs pull
conda env create -p .conda/envs/cuhkx -f environment.yml
```

以下命令均从仓库根目录执行。

## 验证已发布结果

快速 CPU 检查会验证全部已发布 hash、包安全契约、100 MB 限制、保存的 OOF 分数，以及由保存的测试 logits 生成的提交文件：

```bash
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify
```

预期核心结果：

```text
OOF accuracy: 0.9561923583662714
submission rows/classes: 405 / 40
model + detector: 69,819,764 bytes
```

## 从原始测试数据重放推理

将竞赛测试数据放在：

```text
data/Small-Model-Track/Testing/test_file/{test.csv,sample_submission.csv}
data/processed/test/small_model_track_test/SM_test_*/
```

然后运行完整的 YOLO crop、视觉 cache、Skeleton cache 和三分支推理流水线：

```bash
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --output results/strict_v3/reproduced_submission.csv

.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify \
  --replayed-csv results/strict_v3/reproduced_submission.csv
```

重放过程从不读取测试标签、测试集派生统计量、时间戳或用户身份。它会确定性地逐字节复现标准 CSV 的 hash `e2509491…`。建议使用 GPU 推理；`--device cpu` 仅适合较慢的功能性重放。

## 训练

原始数据重放与模型重训练是两项独立检查。上面的命令验证已发布 checkpoint，并不会重新训练它。重训练的准确输入边界、fold、命令和验收条件记录在 [`docs/RETRAINING.zh-CN.md`](docs/RETRAINING.zh-CN.md)（[英文版](docs/RETRAINING.md)）。

训练 cache 由 `data/` 中的模块生成，训练入口位于 `training/`。时序实验使用统一、可审计的 suite runner：

```bash
# 五个 strictV3 fold 加一个 full epoch-5 模型
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.temporal_suite \
  --recipe strict --frame-logits .cache/strict_v3/train_frame_logits.npy \
  --test-frame-logits .cache/strict_v3/test_frame_logits.npy \
  --metadata results/strict_v3/metadata.npz --output runs/strict_v3/temporal

# sched30 实验：三个 seed、五个 fold 和三个 full 模型
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.temporal_suite \
  --recipe sched30 --frame-logits .cache/strict_v3/train_frame_logits.npy \
  --test-frame-logits .cache/strict_v3/test_frame_logits.npy \
  --metadata results/strict_v3/metadata.npz --output runs/experiments/sched30

# Visual R(2+1)D baseline
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.base --help

# Released visual-head family
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.public_finetune --help

# Skeleton/visual fusion and temporal residual head
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.fusion --help
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.training.temporal --help
```

若只执行 train-only 确认性检查，请省略 `--test-frame-logits` 并添加
`--oof-only`。此时 suite 会跳过全部 full-data 任务、拒绝接收 test-logit
参数，并在 receipt 中记录 `test_data_loaded: false`。

按用户划分的 folds 和冻结的发布超参数记录在 `results/strict_v3/release_manifest.json`。不得使用排行榜反馈或匿名测试集属性进行模型选择。

## 重训练状态（2026-09-03）

已从声明的 bootstrap 边界重新运行 temporal、visual-head 和 fusion
折内任务。Temporal 复现了五个历史 fold 准确率；fusion 复现了五个历史最佳准确率；
visual-head 与历史各 fold 的差异不超过 3 个验证样本。独立重训练并重新校准的
`sched30` 候选取得 0.959486 OOF，因此它**没有**替换冻结的 strictV3 发布版本，
也没有替换另行审计的 0.960474 候选包。准确命令、fold 结果和仍待完成的完整部署
验收门禁见 [`docs/RETRAINING.zh-CN.md`](docs/RETRAINING.zh-CN.md)。

另有两项预注册的 train-only 检查被否决。每个 epoch 重新采样相同时序增强后，
三 seed 平均预测仍为 0.937747；对 epoch 3–5 的 TCN 权重做均匀平均后达到
0.938076（3,036 行中净增 1 行），但固定 nested 50/50 发布候选仍精确为
0.960474，预测变化为 0。两项检查均未打开测试数据、未训练 full-data 模型，
也未修改已发布 package。

## 许可证

代码采用 [MIT License](LICENSE)。竞赛数据和预训练权重仍受各自上游条款约束。
