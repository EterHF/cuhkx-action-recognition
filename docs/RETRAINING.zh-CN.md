# 重训练 strictV3

[English](RETRAINING.md) | [简体中文](RETRAINING.zh-CN.md) | [文档索引](README.zh-CN.md) | [仓库首页](../README.zh-CN.md)

## 什么才算重训练

`release.verify` 和 `release.replay` 只验证冻结权重，并不执行训练。一次完整重训练必须从声明的 bootstrap assets 生成新 checkpoint，重新生成 held-user logits，只使用 outer-train 行重新拟合发布校准，最后使用新打包模型执行原始数据推理。

strictV3 是一个 **release-strict** 基线。其公开 R(2+1)D-34 和 DSTFormer bootstrap assets 此前使用 CUHK-X 标签拟合过。因此，0.956192 OOF 是工程重放指标，不是“仅以外部数据初始化的模型”的无偏估计。它与 fully-external 协议的区别保留在技术报告中。

## 固定 folds

| Fold | Held users | 行数 |
| --- | --- | ---: |
| A | 1, 6, 17, 22 | 716 |
| B | 2, 7, 18, 23 | 673 |
| C | 3, 8, 19, 24 | 679 |
| D | 4, 9, 20 | 489 |
| E | 5, 16, 21 | 479 |

Held users 只能用于最终 OOF 测量。某个 held fold 的温度和分支权重拟合只能使用其余四个 fold。

## 输入契约

竞赛数据不会随仓库重新分发。使用 `yolo_r2plus1d.strict_v3.data` 下的模块，从 `data/processed/{train,test}` 构建视觉和 Skeleton cache。训练前必须记录以下输入的 SHA-256 hash：

- `train_depth_ir.npy` 和 `test_depth_ir.npy`；
- `train_skeleton.npy`、`test_skeleton.npy` 及两个 validity mask；
- 来自声明 DSTFormer bootstrap 的 `train_frame_logits.npy` 和 `test_frame_logits.npy`；
- `metadata.npz`、公开 visual bootstrap package，以及任何 fusion resume checkpoint。

标准时序输入的 hash 为：

```text
train_frame_logits.npy  76e6c11c86a681f03f21f7114eae17b1bc8328e6cd6a161f365b8b0cfc37102e
test_frame_logits.npy   b96677c5bc7adc2ef70f53c5819d1042dc8585df2ee4e67965031f10c8db4fbb
metadata.npz            bf2e93e558b4ae148b14835c1f65c943828bff97dfaaf224a39c249c7599f099
```

Suite runner 会将绝对输入路径和 hash 写入 `receipt.json`。因此无需把 checkpoint 文件名误当作来源证明，也能独立审计一次运行。

`data/external/` 明确位于本 strictV3 重训练契约之外。复现标准 0.97512 package
不需要 NTU RGB+D 或 PKU-MMD。完全外部数据研究必须使用独立 manifest，在每个
outer fold 内重新初始化 40 类 head，并将输出保存在 `checkpoints/strict_v3/` 之外。
本仓库不会分发外部数据压缩包或源数据派生的实验权重。这一隔离是来源与发布规则，
并非禁止训练：竞赛主持人[允许公众可获取的外部数据与预训练模型，并明确确认 NTU
RGB+D 可用](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/724404)。
若申请入口对任何人开放，申请制数据也可使用。最终报告必须记录每个外部来源的提供方、
获取步骤、manifest/hash、预处理方式和具体作用。

## 时序基线

运行全部五个 fold 和 full epoch-5 模型：

```bash
CUDA_VISIBLE_DEVICES=0 .conda/envs/cuhkx/bin/python \
  -m yolo_r2plus1d.strict_v3.training.temporal_suite \
  --recipe strict \
  --frame-logits .cache/strict_v3/train_frame_logits.npy \
  --test-frame-logits .cache/strict_v3/test_frame_logits.npy \
  --metadata results/strict_v3/metadata.npz \
  --output runs/strict_v3/temporal
```

历史 fold 准确率 A–E 分别为 0.937151、0.933135、0.945508、0.959100 和 0.926931；aggregate temporal OOF 为 0.940053。即使每个 fold 的准确率一致，不同设备也可能产生并非逐字节相同的 logits。因此，除了 checkpoint hash，还必须比较每 fold 指标和预测差异。

## sched30 实验

该实验固定使用 epoch 5 checkpoint，同时保留 30-epoch cosine schedule。它使用随机种子 2026、2027 和 2028；只有在固定 checkpoint 写入后，验证标签才会被读取一次。

```bash
CUDA_VISIBLE_DEVICES=0 .conda/envs/cuhkx/bin/python \
  -m yolo_r2plus1d.strict_v3.training.temporal_suite \
  --recipe sched30 \
  --frame-logits .cache/strict_v3/train_frame_logits.npy \
  --test-frame-logits .cache/strict_v3/test_frame_logits.npy \
  --metadata results/strict_v3/metadata.npz \
  --output runs/experiments/sched30
```

支持使用 `--device cpu --workers 0` 进行独立数值检查。当 outer-train 分数并列时，CPU 和 A100 运行可能选择相邻的 blend-grid 点，因此部署校准必须同时记录所选权重和设备。候选只有在 aggregate、macro、subject-macro、worst-user 和 worst-fold 指标均不回退，至少四个 fold 不回退，并且原始数据重放与新生成提交一致时才算通过。

若确认性 OOF 筛选不得接触匿名测试资产，请省略 `--test-frame-logits` 并添加
`--oof-only`。该模式会拒绝 test-logit 参数、跳过全部 full-data 任务，将测试路径和
hash 写为 `null`，并在 `receipt.json` 中记录 `test_data_loaded: false`。

## 已执行的重训练审计（截至 2026-09-04）

下列结果来自真实训练，而不是冻结 checkpoint 重放。任务在一张 A100 上串行执行；
当时另一个无关服务占用了该卡的大部分显存。`public_finetune` 和 `fusion` 现已支持
`--cuda-memory-fraction 0.14`，该上限足以运行 visual head-only batch 8 和
fusion batch 8；两个入口也支持 `--deterministic`。历史配方没有启用确定性 cuDNN，
因此复现判据是 fold 准确率和预测差异，而不是 checkpoint hash 完全一致。

| Fold | Temporal 新值 / 历史值 | Visual 新值 / 历史值 | Fusion 新值 / 历史值 |
| --- | ---: | ---: | ---: |
| A | 0.937151 / 0.937151 | 0.960894 / 0.960894 | 0.959497 / 0.959497 |
| B | 0.933135 / 0.933135 | 0.888559 / 0.888559 | 0.974740 / 0.974740 |
| C | 0.945508 / 0.945508 | 0.979381 / 0.983800 | 0.718704 / 0.718704 |
| D | 0.959100 / 0.959100 | 0.860941 / 0.856851 | 0.862986 / 0.862986 |
| E | 0.926931 / 0.926931 | 0.776618 / 0.776618 | 0.874739 / 0.874739 |

Temporal aggregate OOF 为 0.940053。与历史验证预测相比，visual 的 argmax 差异
A–E 分别为 6、0、5、9、4，fusion 分别为 0、1、4、14、0。历史 visual 配方必须
保留 `--workers 8`，因为水平翻转随机数是在 loader worker 内采样的。

独立 CPU `sched30` suite 已完成 3 个 seed × 5 个 fold 及 3 个 full 模型。
其 temporal probability mean OOF 为 0.937747。与冻结的 visual/fusion 分支重新校准后，
候选 OOF 为 0.959486（相对 strictV3 提升 0.003294，5/5 fold 非退化）。该结果低于
另行冻结的 0.960474 候选：CPU/GPU 数值差异使一个 outer-train grid 并列项选择了
相邻权重。因此，本次重训练只证明该方向有潜力，不构成发布晋升。

随后又预注册并执行了三项 train-only 时序泛化检查（3 seeds × 5 folds、CPU、
固定 epoch 5、无测试输入）：

| 候选 | Temporal seed-mean | Fold / seed 稳定性 | 决策 |
| --- | ---: | --- | --- |
| 每 epoch 重采样 reversal/noise | 0.937747（2,847/3,036；无变化） | 5/5 folds、3/3 seeds 非退化 | stage 1 否决：门槛至少为 2,848 行正确 |
| epoch 3–5 FP32 权重均匀平均 | 0.938076（2,848/3,036；+1 行） | 4/5 folds、3/3 seeds 非退化 | 通过 temporal 门禁；release 门禁否决 |
| 固定 0.25 的同类跨用户时序残差混合 | 0.937747（2,847/3,036；top-1 无变化） | 5/5 folds、3/3 seeds 持平 | stage 1 否决：门槛至少为 2,848 行正确 |

权重平均候选的 macro recall 提升 0.000327，subject-macro 提升 0.000208，
worst-user 与 worst-fold 持平。其 train-only nested 分支达到 0.961792
（2,920 行），比历史 nested sched30 分支多 1 行；但预注册的 strictV3 固定
50/50 概率 ensemble 仍与现有候选的逐行预测完全一致：0.960474（2,916 行）、
预测变化为 0，未达到 2,917 行门槛。历史 control 回放精确复现了 nested logits
和最终 probability array。因此停止规则阻止了 full-data 训练、测试推理和打包。

跨用户混合从不同 outer-train subject 中确定性选择同类伙伴，并将其零均值时序残差
以 25% 混入 anchor clip。全部 checkpoint 与 OOF logits 均发生变化，但 seed-mean
预测没有变化；冻结的停止规则禁止继续扫描混合强度或概率。

另行执行的 fully-external NTU 复核在 GPU 上完成了 3 seeds × 5 个 subject folds。
它使用 Kinetics + 25% NTU60 encoder 与全新目标 head：epoch 1 仅训练 head，
epoch 2–15 训练 layer4 + head。这是真正的 scope 变化；历史 head-LR warmup 中，
layer4 从第一步起就保持可训练。单 seed 配对均值为 0.660848，control 为 0.660518
（+0.000329；门槛 +0.002）；各 seed delta 为 −0.010870、+0.009223、+0.002635，
各 seed 同时避免 micro 与 subject-macro 回退的 fold 数仅为 2/4/3。worst-user 平均
delta 为 −0.001517，单格最大下降为 −0.035461。六项冻结判据中五项失败，停止规则
因此禁止 full-data 训练、测试推理、融合、打包和提交。该结果只否决渐进解冻；
PKU-MMD 并不是 treatment 的组成部分，此前的许可疑问已由输入契约中链接的组织方
官方澄清取代。

随后单独完成了 PKU-MMD bridge 的 15/15 个目标域 GPU 作业。唯一 treatment 变量是
seed-matched 外部 encoder：control 使用 Kinetics + 25% NTU60，候选额外经过受约束的
PKU-MMD depth-only layer4 适配，并对三个源域 fold 的 encoder 做 soup。两个 arm 的
全新 head、目标 fold、随机种子、15 epochs、optimizer、temporal-difference 权重 0.10、
BN policy 和固定最终评估均完全一致。

| 冻结 endpoint | Control | PKU bridge | Delta / 门禁 |
| --- | ---: | ---: | --- |
| 各 seed micro 均值 | 0.669521 | 0.677866 | +0.008344；通过 ≥ +0.003 |
| 各 seed subject-macro 均值 | 0.666864 | 0.675394 | +0.008530；通过 ≥ +0.003 |
| 各 seed micro delta（2026 / 2027 / 2028） | — | — | +0.008893 / +0.006917 / +0.009223 |
| 同时满足 micro+subject 非退化的 fold 数 | — | — | 4 / 3 / 4；未达到每个 seed 均 ≥4 |
| fold 单元 worst-user 平均 delta / 单格最大下降 | — | — | +0.015586 / −0.025339；后者上限为 −0.02 |
| train-minus-held gap 平均增量 | — | — | −0.004006；通过 |
| 三 seed logit mean（仅诊断） | 0.676877 | 0.685441 | +0.008564；worst-user 0.43125 → 0.45 |

冻结门禁 10 项通过 8 项，但 fold 稳定性和单格 worst-user 最大下降失败。独立的原始
logit 重算与审计完全一致。因此停止规则禁止 full-data 训练、匿名测试推理、融合、打包
和提交；未读取任何排行榜反馈。

### 经授权的 PKU bridge 部署诊断

上述失败门禁仍是 v4 晋升路线的正式决策。之后另行授权一次 one-shot 诊断，用来回答
更窄的问题：当目标训练准确率仅轻微波动时，理论上泛化更好的外部初始化能否改变公开榜
结果？在读取匿名测试 cache 前，新路线已固定 seed 2028、已经完成训练的 15-epoch
all-train checkpoint、统一量化、strictV3 0.90 / bridge 0.10 概率融合、OOF 阈值、
package 大小和最多一次提交。候选选择期间没有任何排行榜数值可用。

| 冻结阶段 | External OOF | 90/10 融合 OOF | 决策 |
| --- | ---: | ---: | --- |
| FP16 参考 | 2,059/3,036 = 0.678195 | 2,901/3,036 = 0.955534 | 仅作参考；完整发布包超出大小限制 |
| Uniform INT3 v1 | 364/3,036 = 0.119895 | 2,903/3,036 = 0.956192；相对 strictV3 改变 0 行 | 信号塌缩；测试重放未改变提交类别，因此未写 CSV、未提交 |
| 两个混合 INT3/INT4 v2 方案 | 分别正确 355 和 365 行 | 均为 2,903/3,036；相对 strictV3 改变 0 行 | 均未通过 train-only OOF；匿名测试保持未打开 |
| Uniform INT4 v3 | 1,930/3,036 = 0.635705 | 2,900/3,036 = 0.955204；改变 11 行；worst-user 0.8125 | 通过全部冻结 OOF 与部署门禁 |

v3 package 只从嵌套 strictV3 package 中移除了发布权重为 0 的 thermal 分支和两个
冗余顶层 legacy 记录；序列化前后每个可执行 strictV3 对象均逐字节相等。INT4 bridge
package 连同 YOLO 为 99,701,322 bytes。两次 A100 测试重放生成完全相同的 logits
和 CSV；冻结融合改变了 405 个匿名预测中的 1 个（索引 36）。独立审计通过后，唯一一次
Kaggle 提交 `56006027` 得到 **0.97512**，与标准 strictV3 完全持平。记录分数后即
关闭该路线，没有根据排行榜修改权重、量化、epoch 或 seed。

这是一次完整的外部数据*诊断*，但不是完整 strictV3 重训练：标准 visual、fusion 和
temporal 分支没有被替换，外部分支重放读取的是冻结竞赛 cache，而非重新构建全部原始
分支。由于公开榜仅持平而未超过基线，标准 0.97512 package 保持不变。

### NTU60 与 NTU120 规模确认

下一组预注册实验隔离了 NTU 规模和源域覆盖。全部正式模型均使用 seed 2026、五个
subject-wise folds、固定 15 epochs、layer4 + 全新 head、冻结 encoder BN、0.10
temporal-difference 权重和 uniform INT4 部署；任何 fold 都未使用匿名数据。

| 冻结 source / 候选 | FP16 external OOF | INT4 external OOF | strictV3 90/10 INT4 融合 | 决策 |
| --- | ---: | ---: | ---: | --- |
| Kinetics 75% + NTU120 25% | 1,998/3,036 | 1,859/3,036 | 2,903/3,036；改变 10 行 | 允许 full fit 和一次提交 |
| Kinetics 75% + NTU60 12.5% + NTU120 12.5% | 2,029/3,036 | 1,910/3,036 | 2,903/3,036；改变 14 行 | Package 通过，但 CSV 与 ref 56006027 重复；不提交 |
| 上述两行的等权 target-weight soup | — | 1,553/3,036 | 2,902/3,036；改变 3 行 | 未达到预注册 1,900 行 external 门禁；测试前停止 |
| Kinetics 75% + NTU60 25% control | 2,044/3,036 | 1,875/3,036 | 2,902/3,036；改变 13 行 | 未达到预注册 1,900 行 external 门禁；full fit/测试前停止 |

NTU120 full fit 最终训练准确率为 0.962121。其 package 连同 YOLO 为 99,701,322
bytes，两次重放完全一致，只改变匿名索引 133；提交 `56014518` 得到 **0.97512**。
这说明更大的 NTU120 源数据能够经受部署且未损害公开分数，但不能证明仅靠规模就能提升。
NTU60 control 的 FP16 目标 OOF 更高，而更广的源混合比单一 NTU120 更耐 INT4。
因此源域相关性、量化鲁棒性和数据量应作为三个独立变量处理。

## 最终验收

一次完成的 strictV3 重训练必须满足以下全部条件：

1. 为 A–E folds 和完整部署路径生成新的 visual、fusion 与 temporal checkpoint。
2. 生成与 fold 对齐、恰好包含 3,036 行且没有缺失行的 OOF arrays。
3. 校准时排除每个 fold 的标签，并自然产生覆盖 40 类的测试预测集合。
4. 模型连同 YOLO 不超过 100,000,000 bytes。
5. 两次原始数据重放产生相同的 CSV hash。
6. 只有本地门禁通过后才能提交 Kaggle；返回的公开榜分数只记录为外部证据，绝不能用于重新调参。

在全部条件通过前，仓库必须将该运行描述为 partial，并且不得替换标准的 0.97512 发布版本。
