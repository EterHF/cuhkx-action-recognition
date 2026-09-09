# CUHK-X 小模型赛道 — 技术报告

[English](TECHNICAL_REPORT.md) | [简体中文](TECHNICAL_REPORT.zh-CN.md) | [文档索引](README.zh-CN.md) | [仓库首页](../README.zh-CN.md)

**目前最佳 Kaggle 公开榜分数：`0.97512`（第 3 名，ref `55712568`）**<br>
**与第 2 名的差距：`0.00497`（相当于一个公开样本的净变化）。**<br>
**冻结日期：2026-09-04**

> 本报告将历史改进路线和后续策略合并为一份文档。它是项目唯一的顶层技术叙事；逐候选提交日志、strict-v3 优先级备忘录、清理 manifest，以及来自 Codex 任务 `01a04b41…` 的后续方向均汇总于此，不再作为独立文件发布。

---

## 1. 问题与数据

CUHK-X 小模型赛道（[挑战页面](https://openaiotlab.github.io/CUHK-X-Challenge/)，UbiComp / ISWC 2026）是一个包含 40 个类别的跨用户动作识别任务，每个 clip 包含六种模态：

- 16 帧有序的 **Depth Color + IR**（128×128）；YOLO 在 8 帧 IR 上探测并确定统一
  clip 级人体窗口。Thermal 的发布权重为 0。
- 按时间戳排序的 **IMU** 序列，已移除温度、电池和固件元数据。
- **Skeleton**：位置、root motion、相对姿态、速度和加速度。
- **Radar** 帧统计；仅含空 header 的文件按缺失处理。

训练集包含 **3,036 个 clip、18 个用户和 40 个类别**；公开测试集包含 405 条路径。用户级划分是唯一有效的泛化单元——禁止随机按 clip 划分，因为数据中存在 1,043 个近重复 trial group，其中 964 个是三元组。类别不均衡约为 30 倍，每类样本数为 12–365。

原始 EfficientNet-B0 + 230 万参数时序融合头基线在三个互斥用户 fold 上分别得到 53.96%、48.53% 和 50.31%（均值 50.93%，标准差 2.26%）。最终三随机种子 ensemble 为 49.17 MB，远低于小模型赛道 100 MB 限制。

---

## 2. 两种“strict”定义

在整份报告和所有实验中，必须区分“strict”的两种含义：

| 名称 | 定义 | 能支持的结论 |
| --- | --- | --- |
| `release-strict` | strict-v3 发布方案。使用公开 checkpoint；其中部分 checkpoint 在上游训练时见过 CUHK-X 目标标签，因此其预训练可能已包含 outer held-user。 | 可作为当前提交和工程重放基线，**不可**表述为完全无泄漏 OOF。 |
| `fully-external-pretrained strict` | 初始化仅来自 Kinetics、NTU 或 PKU-MMD。每个 CUHK-X outer fold 都重新训练一个 40 类 head，且只使用 outer-train 用户；预处理、epoch 选择和超参数也只由 outer-train 固定。 | 可用于跨用户泛化比较和方法选择。 |

`release-strict` 的 95.619% OOF 是当前提交证据，不是完全无泄漏基线。“research”分支采用 `fully-external-pretrained strict`。

---

## 3. 当前最佳方案：`legal_strict_v3`

`legal_strict_v3` 提交取得 **0.97512**（第 3 名，ref `55712568`）。当前 package 可以从原始测试数据逐字节重建该计分 CSV。关键部署修复是显式设置 `visual_package_output_scale=0.5`：一个保留的 5-bit 视觉成员表示其 0.5 权重贡献，无需存储第二个近重复网络。

| 属性 | 值 |
| --- | --- |
| 官方格式推理 bundle | `checkpoints/strict_v3/submission_bundle.pt`（69,805,793 bytes） |
| 可审计模型源文件 | `checkpoints/strict_v3/model.pt`（64,206,000 bytes） |
| 标准提交 | `results/strict_v3/submission.csv`（405 行、40 类） |
| 原始数据重放提交 | `results/strict_v3/raw_replay/submission.csv`（405 行、40 类） |
| 单 checkpoint 内全部推理权重 | 69,805,793 bytes（余量 30,194,207 bytes） |
| 公开榜分数 | **0.97512**，ref `55712568` |
| 原始数据重放分数 | **0.97512**，与 ref `55712568` 逐字节一致 |
| OOF（release-strict） | aggregate `0.956192`，mean fold `0.954823`，worst fold `0.926931`，macro recall `0.953143` |
| 协议 | `five_fold_subject_wise_nested_temperature_quality_gate`，完整契约见 `release_manifest.json` |
| 原始重放证据 | 确定性 SHA-256 `e2509491…`，与标准 CSV 逐字节一致 |

主持人的[官方 ensemble 澄清](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/729056)
按一个包含全部推理权重（包括每个 ensemble 成员）的 checkpoint 计算大小；磁盘文件必须
**严格小于** 100 MB，允许 FP16、INT8 或更低位量化。因此发布包把原始 YOLO 文件字节
与 strictV3 一起嵌入同一个可用 weights-only 模式加载的 checkpoint，不再依赖此前
“两个文件求和”的解释。

### 3.1 strict-v3 发布包的实际内容

发布模型是一个三分支多分支融合系统，其推理契约记录在 `release_manifest.json`：

* **视觉 head**：仅由训练集拟合的 affine 预处理，不使用逐 clip z-score。
* **TCN / 时序 head**：按固定配方重放每 fold held-out 的 OOF logits。
* **Fusion head**：同样进行重放的 legacy multibranch Fusion4 logits。

第一次清理得到的 package 错误地用两个 4-bit 成员替换了原始 5/6-bit 视觉 ensemble。其确定性重放改变了 `SM_test_0214` 和 `SM_test_0378`，公开榜分数为 0.97014（ref `55978481`）。当前 package 恢复了此前审计过的紧凑表示，在仍远低于大小限制的同时消除了这两处差异。

公开榜最佳历史如下：

| ref | 候选 | 公开榜分数 | 状态 |
| --- | --- | ---: | --- |
| 55613900 | 原始基线 | 0.46268 | 首个基线 |
| 55620179 | YOLO + R(2+1)D-18 | 0.60696 | 首个视觉模型 |
| 55673502 | R(2+1)D-34 int5/int6 + YOLO11n | 0.71641 | 基础视觉候选 |
| 55709862 | Fusion4 / TCN（合法多分支） | 0.89552 | 探索性方案，已否决——package 重放与原始数据重放一致，但公开测试划分泛化较差 |
| 55712568 | **legal_strict_v3** | **0.97512** | **当前最佳** |
| 55714393 | strict adapter v1 / v2 | 0.96517 | 低于 strict-v3，已否决 |
| 55834961 | P0.3 visual mix（base + seed-2027 adapter） | 0.96517 | 低于 strict-v3，已否决 |
| 55846236 | strict seed-pair 2027/2028 P0 | 0.94527 | 已否决——公开测试划分泛化失败 |
| 55857342 | strict top-2-frame-mean temporal pooling | 0.94029 | 已否决 |
| 55858076 | strict top-2 v2 | 0.94029 | 已否决 |
| 56006027 | strictV3 0.90 + PKU bridge INT4 0.10 | **0.97512** | 与标准结果持平；确认性外部数据诊断，不晋升 |
| 56014518 | strictV3 0.90 + NTU120 INT4 0.10 | **0.97512** | 与标准结果持平；更大 NTU 确认，不晋升 |
| 56016293 | strictV3 0.50 + sched30 三随机种子 consensus 0.50 | 0.97014 | 离线 OOF 提升，但公开榜泛化失败；否决 |
| 56035438 | full-fit 等权 Depth Omnivore + IR Omnivore + skeleton，mixed INT8 | 0.58706 | 跨用户泛化严重失败；否决 |
| 56036875 | strictV3 Temporal + Visual 严格 50/50 | 0.95522 | 低于 strictV3；否决 |
| 56036959 | strictV3 Temporal + Visual 继承式门控 | **0.97512** | 与标准持平；更简洁的 private-LB 候选 |

多个离线“更高 OOF”候选（如 OOF 为 0.970* 的 Fusion4 raw）均被公开排行榜否定，不再属于候选方案。

### 3.2 当前最佳方法的数据与训练流程

公开计分的当前最佳仍是 `legal_strict_v3`；train-only OOF 最强候选是它与
`sched30` 的固定 50/50 概率共识。二者共享以下数据流程：

1. 以确定顺序发现 clip，只保留 subject-wise CV 所需的类别与用户 metadata。
2. YOLO11n 在均匀抽取的 8 帧 IR 上探测人体；人体框 union 扩张 1.4 倍，最小边长
   0.35，并按 IR → Depth Color → full frame 回退。
3. 均匀抽取 16 帧对齐的 Depth Color 与 IR，以共享窗口缩放为 128×128，保存 uint8
   `[T,4,H,W]` cache。Affine mean/std 只由相应训练用户拟合，不使用逐 clip 或测试统计量。
4. 构建对齐、以 pelvis 为中心的 H36M-17 skeleton cache，以及 16 帧 crop-scaled
   DSTFormer 输入。Skeleton 缺失只影响 encoder 输入，不作为事后融合 mask。
5. 分别推理视觉 R(2+1)D、视觉+skeleton Fusion、DSTFormer→TCN；temperature calibration
   和 quality gate 只由 outer-train 拟合；full-data 发布权重为 Fusion 0.11、Visual 0.22、Temporal 0.67。

训练使用五个互斥 held-subject folds。Visual head 使用带 0.02 label smoothing 的
cross-entropy 和 train-only horizontal flip；Fusion 使用同一视觉 cache、0.005 skeleton
noise、分离的 visual/new-layer 学习率及 OneCycle schedule。时序 residual TCN 使用
AdamW（3e-4、weight decay 0.01）、0.01 label smoothing、0.25 reversal probability
和 0.01 Gaussian logit noise。`sched30` 在 30-epoch cosine schedule 下固定取 epoch 5，
训练 seeds 2026–2028 后平均概率，再与 strictV3 做固定 50/50 概率均值。它把 release
OOF 从 2,903 提升到 2,916/3,036，5/5 folds 非退化；单 checkpoint 为 98,873,941
bytes，并在 2026-09-04 再次从原始数据复现 CSV SHA-256 `f33e0569…`。它的离线证据更强，
但其冻结 Kaggle 提交仅得 0.97014（ref 56016293），低于 0.97512 公开基线。

### 3.3 复现 `legal_strict_v3`

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --output results/strict_v3/reproduced_submission.csv
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify \
  --replayed-csv results/strict_v3/reproduced_submission.csv
```

验证器要求原始数据重放结果与标准 CSV 逐字节一致。配方记录在 `release_manifest.json`；推理契约的任何变化都必须生成新的 package 和 manifest。

### 3.4 清理后仓库的重训练审计

2026-09-03 使用清理后的训练入口执行了新的参数更新，而非只重放 checkpoint。
strict temporal 分支匹配全部五个历史 fold 准确率，aggregate 为 0.940053；恢复真实的
batch-8 契约后，fusion refinement 匹配全部五个历史最佳 fold 准确率。Visual head
在 A、B、E fold 的准确率完全一致；C、D 分别相差 −3 和 +2 个正确样本。即使准确率
一致，非确定性 cuDNN kernel 与 loader-worker 随机流也会使新 logits 无法逐字节一致。

同时重新训练了三随机种子 `sched30` suite。其 temporal mean OOF 为 0.937747；
与冻结 visual/fusion 分支重新校准后达到 0.959486（相对 strictV3 提升 0.003294，
5/5 fold 非退化）。它与冻结的 0.960474 consensus package 不同：CPU/GPU 数值差异
使一个并列校准项选择了相邻权重。两者都不会改变标准发布版本。在新的 full fusion
package、两次逐字节一致的原始数据重放和经授权的 Kaggle 确认全部通过前，本次重训练
仍是 partial。详细结果和验收标准见 [`RETRAINING.zh-CN.md`](RETRAINING.zh-CN.md)。

---

## 4. 历史改进路线（所有已尝试方向）

本节是完整审计链。每个方向均列出结果、停止原因和标准证据文件。多数具体配方已经
“穷尽”，继续调整同一配方只会扩大验证集搜索空间；但若尚有机制上真正不同、且在结果
揭示前完成预注册的迁移方案，就不能据此宣称某个数据源本身已经穷尽。

### 4.1 原始 EfficientNet-B0 基线（历史）

EfficientNet-B0 多模态基线、辅助损失选择 fold（`pretrained_aux015`，mean OOF ≈ 50.93%）以及三随机种子 ensemble（`pretrained_final_seed{2026,3407,8819}`、`pretrained_ensemble/`）构成了最初的 **release-strict** anchor。这些权重和脚本未随清理后的 main 分支发布，测得结果保留在此以记录来源。

### 4.2 视觉 / 时序 / 融合分支开发（保留为证据）

YOLO crop 的 R(2+1)D-18 → R(2+1)D-34 int5/int6 + YOLO11n 路线将纯视觉候选从 0.46268 提升至 0.71641。加入 ST-GCN/DSTFormer 风格 Skeleton 融合与时序 TCN residual head 后，得到合法的“5-bit multibranch”package。strict-v3 发布版本对这些分支应用按 fold 的 nested selection 规则，不使用排行榜反馈，也不泄漏样本或用户 ID。

### 4.3 外部数据扩展（已测量；重新开放目标域确认）

项目对 NTU Depth / Skeleton 预训练与 PKU-MMD bridge 做了大量探索。最终本地清点
显示：NTU 共 248 GB，包含全部 32 个 masked-depth setup 压缩包和两个 skeleton
压缩包，但 IR 仅有 9 个 setup；PKU-MMD Phase 2 共 156 GB，manifest 含 6,952 个
裁剪后的深度样本、13 个推断 subject 和 41 类。因此，现有数据足以支持诚实的
Depth/Skeleton 研究，但不能宣称完整覆盖 NTU Depth+IR 配对数据。更大的源数据能够
扩展表示覆盖范围，但数据量本身并不构成数学保证：域、模态与标签差异以及灾难性遗忘，
决定了信息能否保留到目标域。本仓库已经观察到明显正迁移，因此不能把数据源本身标记为
“已穷尽”。实测迁移结果如下：

| 实验 | 结果 | 结论 |
| --- | ---: | --- |
| NTU masked-depth 预训练 + R(2+1)D-18，3 seeds × 5 folds | mean micro 63.669%，Kinetics control 60.299%；各 seed 非退化 fold 为 5/4/5 | 通过冻结的迁移门禁；小 backbone 上存在真实正迁移 |
| NTU60 / NTU120 直接监督 R(2+1)D-34 | 58.56% / 58.89% | 大规模源域监督训练会**遗忘** Kinetics 表示 |
| Kinetics anchor + 25% NTU60 R(2+1)D-34 插值，3 seeds × 5 folds | 单格均值 65.760%；matched mean +5.033 pp；15/15 单格非退化 | 目标迁移强且可复现，但 standalone 绝对精度尚不足以发布 |
| 相同 NTU60 插值 + 真正的 epoch-1 head-only 渐进解冻 | 单 seed 配对均值 66.0848%，control 66.0518%（+0.033 pp）；worst-user 均值 −0.152 pp | 未通过预注册的稳定性 / worst-user 门禁；在融合与部署前停止 |
| 仅使用本地 9 个完整 setup 的 NTU Depth+IR | fold C/E 为 63.62% / 61.59%，depth-only 为 68.04% / 61.38% | 部分数据结果混合（C −4.42 pp、E +0.21 pp）；既不能证明也不能否定完整配对 IR 的收益，后者需要单独冻结实验 |
| NTU Skeleton student | source val 64.68%；目标 A–E micro 46.81%、macro 39.51%、worst-user 25.00% | 源域拟合未通过目标域跨用户验证 |
| 早期未约束的 PKU-MMD source-only 预训练 | source val 75.30%，source train 在 epoch 15 约 99% | 该 15-epoch 配方过拟合；源域验证无法代理目标域迁移 |
| 早期 Kinetics → PKU → CUHK-X | 61.92%（control 63.27%） | 直接 continued-training 产生**负迁移**；否决的是配方而非 PKU-MMD 数据 |
| Kinetics+NTU60 → 受约束的 PKU layer4 bridge，源域 subject CV | 63.828%，control 57.892%（+5.936 pp）；9/9 seed-fold 单元为正 | 通过全部冻结的源域门禁，足以支持一次独立目标域确认 |
| 同一 bridge 的目标域 3 seeds × 5 folds | mean micro 67.7866%，control 66.9521%（+0.8344 pp）；每个 seed 均为正；诊断 logit mean 68.5441% | 目标迁移真实存在，但 seed 2027 仅 3/5 fold 稳定，且一个 worst-user 单元下降 2.5339 pp，故否决晋升 |
| 固定 seed-2028 bridge、uniform INT4、strictV3 90/10 部署诊断 | external OOF 63.5705%；融合 OOF 95.5204%；Kaggle 0.97512（ref 56006027） | 通过诊断门禁并与标准结果持平；无精度增益，不晋升、不根据榜单调参 |
| 用 PKU visual 替换 strict-v3 visual | 95.191%（control 95.619%）；nested mean weight = 0 | 不能作为第 4 个 logit 分支 |

渐进解冻实验是在已经有用的 NTU 初始化上改变目标训练范围，并没有检验“NTU 数据是否
有用”。同样，早期 PKU 失败采用的是约束较弱的不同配方。§5.7 的 seed-matched PKU
bridge 确认因此只替换外部初始化，目标域配方与不可变 matched control 完全一致。

### 4.4 Head-only / L2-SP / SAM / EMA / temporal-diff（已穷尽）

这些机制都在完整 A–E 矩阵上通过 release-strict inner CV 做了全面测试：

| 方向 | 结果 | 参考 |
| --- | --- | --- |
| Head-only fine-tune（完整 A–E） | micro 0.146684——严重欠拟合 | `audit_head_only_scope_full_ae_v1.py` |
| L2-SP λ = 0.01（完整 A–E） | micro 0.664141，delta −0.005——不发布 | `audit_l2sp_temporal_difference_full_ae_v1.py` |
| Canonical SAM ρ = 0.05（C/E） | micro 0.585780，delta −0.076——不扩展 | `audit_sam_temporal010_screen_v1.py` |
| Success-step EMA 0.995（C/E） | micro 0.662061——不扩展 | `audit_success_step_ema_temporal010_v1.py` |
| Temporal-difference w = 0.10（完整 A–E） | micro 0.669521——未通过 all-seed、worst-user 和 generalisation-gap 门禁 | `audit_temporal_difference_weight010_v1.py` |

以上方向均未晋升。

### 4.5 Multirate / R2D18 / residual / 历史 TSM / category-pair / TTA / calibration / margin-arbitration / quantization-sweeps（已穷尽）

这些方向均被 strict inner CV 否决：

* **Multirate（连续 Skeleton + IMU）分支融合**：任意小权重都会使 fold C 回退；fold E 最多提升约 1%，但违反双 fold 非退化规则。永久停止。
* **Strict R2D18 单模态 ensemble**：三随机种子稳定性失败；每个 fold 都比基线多错 1–6 个 OOF 样本。
* **Residual / boosted residual 多窗口融合**：fold C 仍回退 3–4 行，瓶颈在表示能力。
* **历史 TSM-MobileNet C/E（固定 epoch 3）**：0.280 / 0.232，远低于可用的互补区间，不再训练。
* **Nested top-2 category-pair correction**：inner selection 将其收缩到零变化；held/validation tag 泄漏使其无法形成有效信号。
* **固定 margin arbitration / calibration / TTA / 多随机种子量化 sweep**：全部收敛到同一个失败 CSV；最高 OOF 为 0.95883，低于 strict-v3 候选。该路径已穷尽。

### 4.6 合规与规则边界（保留为证据）

* **LLM 使用规则**：仅禁止在预测阶段使用 LLM，允许 AI 编程助手。该结论经 Kaggle discussion 回复确认，并在清理前保留于历史审计。
* **竞赛外部数据规则**：组织方的[官方澄清](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/724404)
  允许使用公众可免费或经合理步骤获取的外部数据集与预训练模型，但必须在最终报告中披露。
  任何人都能提交的申请表也符合要求，并且 NTU RGB+D 被明确点名允许。这一证据取代了
  后续本地审查中“许可尚未解决”的错误结论；对应的主持人回复为 `3495517`
  （2026-07-12）、`3496001`（2026-07-13）和 `3504934`（2026-07-29）。
* **NTU RGB+D 条款**：[官方提供页面](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
  将数据用途限制为学术研究，并限制再分发和商业使用。本地压缩包及 NTU 派生研究
  checkpoint 均保持在开源发布包之外。
* **PKU-MMD 许可证**：[官方项目页面](https://struct002.github.io/PKUMMD/)
  直接公开了研究数据，但没有单列明确的数据许可证。报告保留这一剩余限制，且不再分发
  原始数据；但不能把它误写成竞赛训练禁令。任何拟公开发布的派生权重仍须另行通过
  clean-release 审查。
* **预训练与蒸馏**：组织方的[官方回复](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/711665)
  允许 ResNet-18 一类小型标准预训练模型，也允许知识蒸馏。MViTv2-S 约 131.9 MB，
  不能在 ≤100 MB 规则下直接作为最终 Small-Track artifact；但可作为仅训练阶段的 teacher
  研究，前提是提交的 student 与完整 package 满足最终模型规则。
* **TorchVision 许可证提醒**：预训练权重可能继承其训练数据条款，已记录在 External-Compliance 审查中。
* **决赛开源许可证**：[赛事官方页面](https://openaiotlab.github.io/CUHK-X-Challenge/)
  要求 Top-6 决赛方案在决赛后 30 天内以 Apache-2.0 发布；本仓库当前采用 MIT。
  若进入决赛，必须由拥有相关版权的维护者确认并执行许可证迁移；自动整理过程不能静默
  改写第三方或其他贡献者的权利。

---

## 5. 探索记录与冻结的后续项

这些方向构成归档研究队列。它们记录在此，以确保清理后仍能保留设计依据；其 preregistration、实验和协议代码有意不随 strictV3-only main 分支发布。

### 5.1 候选融合的严格 nested selection（优先级 1）

* **动机**：所有手工选择的系数和由排行榜驱动的单行修改都已被否定。唯一可接受的选择规则是：“每个 held-fold 的系数只能由其余四个 fold 的 OOF 决定，不得使用用户/样本 ID 或测试集属性”。
* **已实现（CPU / synthetic）**：Fusion7 nested selection（OOF 净增 14 行、5/5 非退化、自然覆盖 40 类）；50/50 probability-mean 对 strict-v3（5/5 非退化、净增 9 行、距离接收标准差 1 行）；nested coefficient grid（端点及 0.25 / 0.5 / 0.75）；shared-state multi-pooling 候选（energy + top-2，共用 Fusion4 / Visual4 / DSTFormer，OOF 净增 12 行、5/5 非退化、自然覆盖 40 类；历史上按组件估算为 86.43 MB、连同 YOLO 为 92.05 MB，但这不是官方单 checkpoint 实测值）。
* **2026-09-04 收口**：最终 `sched30` package 已生成一个 98,873,941-byte checkpoint，
  并通过其中嵌入的 YOLO 字节从原始数据重放；CSV 与冻结候选逐字节相同（`f33e0569…`）。
  随后的冻结 Kaggle 提交得 0.97014（ref 56016293），因此不予晋升。更广泛的历史
  Fusion7 精度差异只保留为归档方向，不作为发布结论。
* **后续设计要求**：
  1. selector 只能读取其余四个 fold 的 OOF logits、marginal 和 agreement 信号。禁止使用用户或样本级属性。
  2. 选择规则必须预先声明，不允许“查看 fold C 后再调整 threshold”。
  3. 只有在两轮原始数据重放一致且完整 nested gate 通过后，候选才能加入提交队列。

### 5.2 等权概率 consensus / multi-pooling（优先级 2）

* **动机**：量化、校准和 TTA 路径均已穷尽后，“相同权重、不同 pooling”仍可修正少量错误且不增加新权重，能够维持 100 MB 预算。
* **已实现**：
  * `sched30` 三随机种子时序 consensus（FP32）：OOF +0.004282、5/5 非退化、每个 seed 均为 5/5；单 checkpoint 为 98.87 MB；CSV hash 为 `f33e0569…`；Kaggle public 0.97014（ref 56016293），不予晋升。
  * 时序 pooling 多数投票（top-2 + energy）：OOF +0.003623、5/5 非退化；单 checkpoint 为 96.94 MB；CSV hash 为 `6a320486…`。
  * strictV3、`sched30`、temporal pooling 的固定三方多数投票仅为 2,909/3,036，
    低于 `sched30` 的 2,916/3,036。基于 confidence/margin/entropy 的 nested routing
    和类别先验校正也未超过固定共识，因此没有继续测试推理。
* **后续设计要求**：
  1. 将两个 pooling variant 视为结构上的最小集合，并执行严格 OOF nested selection，不能使用固定 50/50 融合，也不能手工挑选系数。
  2. 只有与 anchor / `sched30` 共享 DSTFormer frame logits 且不增加大型权重的新 pooling variant 才可接收。
  3. 每个候选必须生成两份逐字节相等的 CSV，并确保 deployment logits 与 training OOF 在 FP16/FP32 下精确匹配。

### 5.3 从零训练的小 backbone——“仅竞赛使用”（优先级 3，多样性 fallback）

* **动机**：从零训练的小模型提供真正不同的归纳偏置和来源清晰的 fallback。它不是因为
  “外部训练数据被禁止”才需要；官方已经允许外部数据。最终部署模型仍须满足小模型赛道
  ≤100 MB 规则，而 MViTv2-S 自身过大，不能直接作为最终 artifact。
* **候选**（已完成 preregistration 和 CPU / contract 审计，87/87 contract 与 20/20 signal 测试均通过）：
  * **主候选**：`TSM-MobileNetV3-Small`，975,576 个参数，FP32 `state_dict` 约 4.03 MB；显式使用 `weights=None`，不进行外部下载。比较 `no_shift_meanmax`（C0）与 `tsm_meanmax(div=8)`（C1），两个 arm 在每个 seed 使用完全相同的逐张量初始化。
  * **备用候选**：`S3D`，7,954,184 个参数，FP32 约 32.07 MB，同样使用 `weights=None`。
* **冻结配方（不调参、不按 held 结果选择 epoch）**：
  * TSM：100 epochs，SGD momentum 0.9，LR 0.01（目标 effective batch 64），weight decay 1e-4，LR 在 40/80 epoch 衰减。模型 head、BN、mean+max、edge-keeping 和 cache augmentation 均视为本项目 adapter 状态，**不得声称这是论文配方**。
  * S3D：只使用一个固定配置，不做 grid search。
* **后续设计要求**：
  1. **单进程配对训练**：一个 job 使用同一个增强 batch 训练 C0/C1；构造 held loader 前执行 delete→rebuild 后 reload；held loader 只能迭代一次。
  2. 候选空间冻结为 `C0/C1/C2`；不得在看到 TSM 结果后再加入 S3D。
  3. 划分为 2 个 inner seed × 4 个 inner fold（约 80 个 process job），随后 outer 最多执行 30 个 one-shot job；每个 job 的审计必须为 P0/P1 = 0。
  4. Outer validation 使用按 fold 的本地选择。只有 5/5 outer fold 一致选择同一方向时，候选才允许晋升为新的 outer preregistration。
  5. 共享 builder 必须在 build→audit→launch 的独占 single-writer 窗口中运行；TSM runner 不得向 outer 泄漏 C1 指标。
  6. 输入统一为 **input snapshot + SHA manifest**：`authorization` 至少 21 行；contract 比较只使用逻辑路径，I/O 只能通过 private snapshot。

### 5.4 外部表示迁移（按冻结门禁执行中）

合规纠正后，受控的外部数据工作重新开放，但不能借此重启事后超参数搜索。队列如下：

1. **PKU-MMD bridge 目标域确认（已在 §5.7 完成并否决）**：只比较一次
   seed-matched 初始化与不可变 control；平均迁移为正，但合取稳定性门禁失败。禁止扫描
   源权重、LR 或 epoch。
2. **VideoMAE-S ≈ MViTv2-S**：“≈”表示相同调查优先级，而非二者在本数据集等价。在打开任何 CUHK-X fold 前冻结外部 checkpoint、架构、预处理和 seed map。
3. **本报告关闭外部 IR**：NTU 本地只有 9/32 个完整 IR setup；PKU-MMD 本地虽有
   902,397 张 depth PNG，却没有 IR 归档或展开后的 IR 树。补齐 NTU 还需 320.8 GB，
   已停止下载。本报告不使用不完整配对 IR，外部迁移结论均明确为 depth-only。
4. **确定性 depth / lag-1 temporal-difference channel**：配方固定、可审计，并预先声明辅助权重；不得根据 held 结果搜索 channel recipe。
5. **预声明的尾部权重平均（已执行并否决）**：在打开 A–E 前声明固定 epoch 3–5 区间、LR schedule、参数范围和 BN 处理方式；平均后的 checkpoint 是唯一候选，结果冻结于 §5.5，禁止事后扫描其他 SWA schedule。
6. **SlowFast / X3D**：沿用相同 subject folds、seed budget 和 source-only 数据边界；先在本地测量计算量和准确率。
7. **MixStyle / ASAM**：排在最后，因为小 fold 上的偶然增益往往损害跨 seed 稳定性。

External-only 的 strict 定义是：外部表示、公开架构和确定性训练变换。**绝不能**使用测试/匿名标签、ID、样本、提交分数、预测历史或事后排行榜反馈。

### 5.5 2026-09-03 时序泛化检查（已完成并否决）

三个单变量候选都在执行 A–E 前完成锁定。三者使用相同的训练 frame-logit SHA
`76e6c11…7102e`、metadata SHA `bf2e93e5…9f099`、随机种子
2026/2027/2028、五个固定 sched30 epochs，并延迟读取 held 指标。Suite 以
OOF-only 模式分别运行：每个候选 15/15 jobs 完成，未生成 full-data 模型或测试文件，每份
receipt 都记录 `test_data_loaded=false`。

| 候选 | 相对 fresh sched30 的 temporal 结果 | 泛化门禁 | Release 门禁 / 决策 |
| --- | --- | --- | --- |
| 每 epoch 确定性重采样 reversal/noise | 2,847/3,036 = 0.937747；seed 2026 多 1 行，但 seed mean 的预测变化为 0 | macro、subject-macro、worst-user、worst-fold 及 5/5 folds 持平；未达到预注册的净增 1 行门槛 | stage 1 停止并否决 |
| 对 post-update epoch 3–5 的 FP32 TCN 参数做均匀平均 | 2,848/3,036 = 0.938076（+1 行）；macro +0.000327；subject-macro +0.000208；worst 指标持平 | 4/5 folds、3/3 matching seeds 非退化；通过 stage 1 | nested 分支为 2,920/3,036 = 0.961792（比历史分支 +1 行），但固定 50/50 strictV3 consensus 仍为 2,916/3,036 = 0.960474，预测变化为 0；门槛为 2,917，因此在 stage 2 否决 |
| 固定 0.25 的同类跨用户时序残差混合 | 2,847/3,036 = 0.937747；三个 seed 和 5/5 fold 分数均持平，top-1 变化为 0 | macro、subject-macro、worst-user、worst-fold 持平，但未达到净增 1 行门槛 | stage 1 停止并否决；不扫描强度或概率 |

train-only nested evaluator 先用历史 sched30 输入做控制回放：保存的 nested logits
与最终 probability array 均逐元素相等（`max_abs=0`）。证据 hash 为：epoch 重采样
receipt `dd135d65…df24`、OOF `0ddb3bdd…67c0`；尾部权重平均 receipt
`40510e90…26ac`、OOF `99a0df0b…5960`；最终 nested-gate metrics
`d8edb37a…044b2`。对于本轮数据/训练候选，每个 outer-train 折的每个类别都至少有
两个用户；确定性选择同类、不同 outer-train 用户的伙伴，将其零均值 16 帧残差以 25%
混入，同时保持 anchor clip 均值不变。15 个 checkpoint 全部不同于 control，OOF logits
也确实变化（`max_abs=0.115523`），证明 treatment 已执行，但 3,036 个决策无一改变。
其 receipt 为 `8f1e8612…787b`、OOF 为 `1cb107ed…6a12`、否决 decision 为
`7e20e00e…a01a`。

停止规则禁止 full-data 训练、匿名测试推理、打包，以及在看到 A–E 后修改系数。按照
仓库清理策略，否决机制仅保留在本报告中，不成为长期训练 flag 或模型文件。

### 5.6 2026-09-03 NTU 渐进解冻复核（已完成并否决）

历史 `head_warmup_epochs=2` 网格只对新 head 的学习率做爬升，**并未**冻结 layer4。
因此，本轮在训练前锁定了一个真正不同的变量：R(2+1)D-34 从冻结的
Kinetics + 25% NTU60 masked-depth encoder 初始化；epoch 1 仅训练新的 40 类 head，
epoch 2–15 再训练 layer4 + head。Cache、按 fold 的 affine contract、optimizer group、
cosine schedule、augmentation、BN policy、fold、seed 和固定最终 checkpoint 选择均与
matched control 相同。

全部 15 个 GPU 作业（3 seeds × 5 个 subject folds）完成。Runner 断言 epoch 1 的
layer4 逐 bit 不变而 head 确实改变，并断言解冻后 layer4 确实更新。每个 epoch-15
checkpoint 都先写盘并重新加载，之后才唯一一次评估 held 标签。

| 冻结 endpoint | Control | 渐进解冻 | Delta / 门禁 |
| --- | ---: | ---: | --- |
| 单 seed 配对 micro 均值 | 0.660518 | 0.660848 | +0.000329；要求 ≥ +0.002 |
| subject-macro 均值 | 0.653176 | 0.654989 | +0.001813；仅该方向通过 |
| 各 seed micro delta（2026 / 2027 / 2028） | — | — | −0.010870 / +0.009223 / +0.002635 |
| 各 seed 同时满足 micro+subject 非退化的 fold 数 | — | — | 2 / 4 / 3；要求每个 seed 均 ≥ 4 |
| worst-user 平均 delta / 单格最大下降 | — | — | −0.001517 / −0.035461；两项均失败 |
| 三 seed logit mean（仅诊断，不是门禁） | 0.669960（2,034 行） | 0.672596（2,042 行） | +8 行，但 worst-user 从 0.44375 降至 0.425 |

六项预注册判据中，仅 subject-macro 方向通过，其余五项失败。因此，该候选被否决；
不扫描 warmup 长度或 LR，不执行 strictV3 融合、full-data 训练、匿名测试访问、
checkpoint 打包或 Kaggle 提交。冻结证据标识为：preregistration
`1b21f4ef…ddadc`、runner `c923a4d9…f556`、summary `ee2fc114…49380`、decision
`57776a38…f949bd`、seed-mean OOF logits `a339f104…3cb10`。这一结果只否决渐进解冻，
PKU-MMD 并不是该 treatment 的组成部分。此前“其竞赛许可尚未解决”的表述有误，现由
§4.6 中的组织方官方澄清取代。

### 5.7 2026-09-04 PKU-MMD bridge 目标域确认（正迁移；晋升否决）

组织方澄清以及 2026-08-30 已批准的访问审查，使此前 materialise 的 PKU-MMD bridge
获得一次全新的目标域确认。执行前没有打开任何未完成或隔离的历史目标结果；15 个作业、
全部输入 hash 和合取门禁都在训练前冻结。唯一 treatment 变量是 encoder 初始化：

* control：Kinetics anchor + 25% NTU60 masked-depth 插值；
* treatment：相同 anchor，再在三个源域 subject folds 上执行六个固定 epoch 的 PKU-MMD
  depth-only layer4 适配，并按 seed 对三个 fold 的 encoder 做等权 source soup。

两个 arm 在对应 seed 使用逐 bit 相同的全新 40 类 head，并共享 A–E 目标 folds、
temporal-difference 权重 0.10、layer4+head 范围、optimizer、LR schedule、冻结 BN、
15 个固定 epoch，以及 checkpoint 重载后的唯一一次 held 评估。Runner 证明目标训练前
只有 encoder 条目不同。全部 15 个作业在 GPU 0/1 完成，期间没有提前打开 aggregate。

| 冻结 endpoint | Control | PKU bridge | Delta / 门禁 |
| --- | ---: | ---: | --- |
| 各 seed micro 均值 | 0.669521 | 0.677866 | +0.008344；通过 ≥ +0.003 |
| 各 seed subject-macro 均值 | 0.666864 | 0.675394 | +0.008530；通过 ≥ +0.003 |
| 各 seed micro delta（2026 / 2027 / 2028） | — | — | +0.008893 / +0.006917 / +0.009223；全部通过 |
| 各 seed subject-macro delta | — | — | +0.010067 / +0.005704 / +0.009819；全部通过 |
| 同时满足 micro+subject 非退化的 fold 数 | — | — | 4 / 3 / 4；未达到每个 seed 均 ≥4 |
| fold 单元 worst-user 平均 delta | — | — | +0.015586；通过 |
| 单格 worst-user 最大下降 | — | — | −0.025339；未通过 −0.02 上限 |
| 候选 micro / subject 的 seed 标准差 | — | 0.004456 / 0.005086 | 均通过 ≤0.01 |
| train-minus-held gap 平均增量 | — | — | −0.004006；通过 ≤0.01 |
| 三 seed logit mean（仅诊断，不是门禁） | 0.676877 | 0.685441 | +0.008564；worst-user 0.43125 → 0.45 |

这里必须区分两个结论：受约束地加入 PKU-MMD 信息后，目标域的平均收益真实且稳定，
三个 seed 的 micro 与 subject-macro 都为正，因此数据假设成立；但预注册晋升规则不仅
要求均值提高，还要求局部 fold 鲁棒性。seed 2027 的 D/E 回退，且 C 的 worst-user
下降 2.5339 pp，最终 10 项门禁通过 8 项。看到结果后没有放宽阈值。

独立程序从 15 对候选/control 原始 logits 重新计算，全部 aggregate 与审计精确一致。
冻结标识为：source materialization `4110434a…153f`、preregistration
`a4dc86f0…aa00`、runner `49fcf679…dc14`、auditor `9cd7f284…60ba`、summary
`682223ca…3313`、decision `71625074…8243`、seed-mean OOF logits
`741cfd38…1a02`。全程未读取匿名/测试资产、提交或排行榜反馈。合取门禁失败后，停止
full-data 训练、融合、打包和提交；外部数据的正向证据则保留给下一项机制上真正不同、
重新预注册的实验。

### 5.8 2026-09-04 one-shot INT4 PKU 部署诊断（已完成；持平）

§5.7 之后，用户明确授权对训练充分的 NTU/PKU 模型执行一次独立排行榜诊断，即使目标
OOF 仅有轻微波动。这并未重新打开 v4 已失败的阈值。新的 preregistration 固定一个
seed-2028 checkpoint、15 个 all-train epochs、layer4 + 全新 head 训练范围、
temporal-difference 权重 0.10、水平翻转 TTA，以及 strictV3 0.90 / bridge 0.10
概率融合。候选选择仅使用训练侧；早先 v1 的匿名输出被明确禁止作为 v2/v3 设计输入。

部署精度而非 GPU 显存成为主要限制。完整 FP16 bridge 的 OOF 为 2,059/3,036
（0.678195），但无法与 strictV3、YOLO 同时满足大小限制。Uniform INT3 塌缩至
364 行；两个预注册的混合 INT3/INT4 修复分别塌缩至 355 和 365 行，且均未打开匿名
测试数据。Train-only INT4 probe 保留了有效信号，因此 v3 只冻结一个 uniform
per-output-channel signed INT4 候选。

| v3 冻结 endpoint | 结果 | 门禁 |
| --- | ---: | --- |
| INT4 external OOF | 1,930/3,036 = 0.635705；subject-macro 0.633421 | 通过 ≥1,800 行 |
| 90/10 概率融合 OOF | 2,900/3,036 = 0.955204；worst-user 0.8125 | 通过 ≥2,898 行且 worst-user ≥0.8125 |
| 相对标准结果改变的 OOF top-1 | 11 | 通过 ≥1 |
| 含 YOLO 的单 checkpoint | 99,978,253 bytes | 通过 <100,000,000 |
| 匿名重放 | 两次 package-backed GPU logits 与两份 CSV 逐字节一致；改变 1 个预测（索引 36） | 通过 |
| Kaggle 确认 | **0.97512**，ref `56006027` | 与标准 strictV3 完全持平 |

大小修复没有量化或以其他方式改变 strictV3。它只移除了 release weight 精确为 0 的
MobileNet thermal 分支，以及冗余的顶层 legacy `blend`、`yolo_bytes` 记录。
Fusion、visual、DSTFormer、temporal residual 和可执行 release contract 均经过递归
比较，并在安全的 `weights_only=True` 重载后保持逐字节相等。最终 package 使用
legacy pickle protocol 2，因为正式 OOF 前的兼容性自检发现 PyTorch 2.6 的
weights-only loader 会拒绝 protocol 4 的 opcode 149；该修正在 formal OOF 前已记录。

独立审计重新计算 90/10 概率与提交文件，确认全部 63,464,372 个量化权重均为 INT4，
并匹配冻结 package hash。上传前配额为剩余 5 次，上传后为 4 次。公开分数只在提交完成
后才被看到，并未用于改变候选。冻结标识为：preregistration `44661e56…c701`、runner
`c93c311b…b10c`、OOF summary `4a85db19…a60e`、decision `da23595b…fd76`、
package `054c9e46…c750`、auditor `08780802…b3d`、audit record
`8a671464…12b3`、submission CSV `b52ebd13…1869`。

该结果在有限意义上支持用户假设：更大数据初始化在 INT4 部署后仍保留有效信号，并且
没有降低公开榜准确率；但它也没有提高准确率。由于只改变一个匿名预测且分数持平，
strictV3 仍是更简单的标准发布版本；v3 bridge 仅作为报告证据保留，不进入 main。

### 5.9 NTU 规模与覆盖实验（已完成；不晋升）

规模实验在相同 seed-2026、5-fold、15-epoch、layer4+head 目标协议下比较固定的
Kinetics-anchor NTU 初始化。Kinetics+25% NTU120 的 external OOF 在 FP16 和 INT4
下分别正确 1,998 和 1,859 行；冻结的 strictV3 90/10 融合保持 2,903/3,036、
worst-user 0.8125，并改变 10 个 OOF 决策。Full fit 最终训练准确率为 0.962121。
99,971,501-byte 单 checkpoint 在 A100 上两次重放完全一致，将测试索引 133 从类别 24 改为
19；Kaggle ref `56014518` 得到 **0.97512**。

NTU60/NTU120 等权源更新把 INT4 external OOF 提高到 1,910 行，融合改变 14 个 OOF
预测且仍正确 2,903 行，但最终 CSV 与此前 PKU 诊断逐字节相同（`b52ebd13…`），因此
新颖性门禁阻止重复上传。两个目标模型在权重空间等权平均后，INT4 external OOF
塌缩到 1,553 行，在测试访问前失败。最后，确定性重训练的 NTU60 control 复现了
2,044 行 FP16 正确，但 INT4 仅 1,875 行，低于冻结的 1,900 行门槛；同样在 full-data
训练和测试推理前停止。

这些证据否定“源数据更多就必然在受限部署中单调提升”的简单结论。NTU120 增加动作
覆盖，NTU60 在 FP16 下仍更贴近目标域，混合源则比单一 NTU120 更耐 INT4。没有任何
候选超过标准 strictV3，因此 main 发布保持不变。当天主动保留了 3 次提交配额，因为
没有其他候选同时通过证据和新颖性门禁。

### 5.10 快速收口：打包规则与 NTU 模态清点

2026-09-04 的收口检查在不打开新目标 fold 的前提下核对了最后两个假设。主持人的
ensemble 裁定要求一个 checkpoint，不能只把多个独立文件的大小相加。实际单文件
materialisation 结果为：

| 推理 artifact | 单 checkpoint bytes | 距 100,000,000 的余量 |
| --- | ---: | ---: |
| 标准 strictV3 | 69,805,793 | 30,194,207 |
| PKU INT4 诊断 | 99,978,253 | 21,747 |
| NTU120 INT4 诊断 | 99,971,501 | 28,499 |
| NTU60/120 INT4 诊断 | 99,975,005 | 24,995 |

四者均通过，但三个实验 artifact 的真实余量只有 21–28 kB。因此后续必须以最终序列化
文件执行门禁，不再用组件大小做算术推断。公开 strictV3 验证器现在强制检查
69,805,793-byte bundle 及其中嵌入 detector 的 hash。

[数据提供方清单](https://rose1.ntu.edu.sg/dataset/actionRecognition/)说明 NTU RGB+D
120 共 114,480 个样本，提供 masked depth、skeleton 与 IR；masked depth 共 147 GB，
IR 共 389 GB。经登录后的官方 endpoint 尺寸与本地 32 个 masked-depth 压缩包全部
一致，两个 skeleton 包也已存在；IR 只有 9/32 setups（完整包共 97,333,511,077 bytes），
缺少的 23 个官方包共 320,783,737,075 bytes。曾测试可续传直连下载，但经资源/收益
复核后已经停止，目前没有下载进程。另行检查的 PKU-MMD Phase 2 本地树具有完整三视角
depth 帧，但没有可用 IR。因此本报告关闭外部 IR，不在不完整子集上训练；该问题也没有
下载 RGB 或 full depth。

### 5.11 2026-09-04 非 IR 快速 trick（已完成）

执行了两项有界筛选。首先，strictV3、`sched30_consensus` 与
`temporal_pool_consensus` 的多数投票在不引入回退的情况下修复 6 个 strictV3 错误，
但只有 2,909/3,036，低于已冻结 `sched30` 的 2,916/3,036；confidence、margin、
entropy 及类别先验 nested routing 同样没有提升固定共识。

其次，train-only subject-balanced sampling 在不增加推理权重的前提下直接针对跨用户
鲁棒性。第一轮比较因历史 control 的 loader-worker 随机流不同，在汇报前作废。替代的
严格配对运行固定 seed 2026、workers=4、确定性 kernel、一个 head-only epoch、预处理
契约和其他全部参数，唯一变化是 `WeightedRandomSampler`。A–E delta 分别为 −0.279、
−0.892、−0.295、+1.636、−1.461 pp；均值从 89.1222% 降到 88.8641%（−0.2582 pp），
仅 1/5 folds 非退化，worst fold 下降 1.4614 pp。该方向未访问测试集，也没有扫描第二个
采样权重，按门禁否决。固定等概率 `sched30` consensus 是唯一通过离线门禁的新小 trick，
但其 0.97014 公开分数不支持晋升。

### 5.12 分支冗余与原生帧率 cross-attention，2026-09-05

恢复的 strictV3 15 个分支数组首先全部通过 release manifest 中记录的 SHA-256。
subject-wise OOF 表明 Temporal 是主模型，Visual 提供有效多样性，而 Fusion 是最弱的
独立编码器：

| 分支 | 正确数 / 3,036 | Accuracy | Subject-macro | Worst user | 其余两支都错时独立答对 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fusion | 2,430 | 0.800395 | 0.795598 | 0.478528 | 8 |
| Visual | 2,723 | 0.896904 | 0.900112 | 0.618750 | 30 |
| Temporal | 2,854 | 0.940053 | 0.939254 | 0.812500 | 61 |

保持冻结的逐 fold temperature 和 gate，只将一个分支权重置零的诊断结果为：
Temporal+Visual 2,909 行、Temporal+Fusion 2,867 行、Visual+Fusion 2,766 行，完整三支为
2,903 行。该结果属于事后结构诊断，不能当作可晋升 OOF 候选；但它支持在未来模型中删除
重复的 Fusion R(2+1)D 编码器。

随后生成的新 cache 在256帧 padding 上限内保留每个有序原生 Skeleton 帧。帧数
median/p95/max 为23/69/236；没有 clip 被截断，没有插值或重复，共保留85,879个有效
Skeleton 帧。单次 R(2+1)D-34 前向提供8个 layer-2 visual token。一个135,465参数的 head
以 pose 和相邻帧 velocity 为输入，通过无 stride 局部卷积编码，以 Skeleton 帧为 query、
visual token 为 key/value 执行四头 cross-attention，并对冻结基线添加零初始化 residual。

固定12 epoch 的 v1 将 Temporal 从2,854降到2,844行（修复18、破坏28）。唯一一次预注册
的保守后续以无 Fusion 的 Temporal+Visual 为基线，固定 residual scale 0.25 和 teacher-KL
1.0，结果从2,909降到2,904行，各 fold delta 为0/−1/−3/0/−1。两版均失败；未运行测试
推理，也没有继续扫描 scale/KL。因此结论必须拆开：删除 Fusion 有证据支持，但当前两种
高频 cross-attention 训练配方没有得到支持。

### 5.13 独立公共 Depth/IR 分支，2026-09-05

后续实验使用分别微调的公共 Omnivore Swin-T 替换旧 Visual/Fusion 编码器，同时保留
原生帧率 skeleton 编码器；没有加载历史项目 checkpoint。在固定 716 行跨用户 holdout
上，独立 Depth、IR、skeleton 的 accuracy 分别为 0.472067、0.544693、0.455307，
worst-user 分别为 0.372549、0.418301、0.359477。

Cross-attention 没有提升该证据：两个整段传感器 token 得到 0.539106/0.411765
（accuracy/worst-user）；每个传感器保留 8 个时间 token 后仅为 0.515363/0.392157，
尽管训练 accuracy 达到 0.9651，故该 learned fusion head 被判定为过拟合。三个独立监督
分支固定等权 logits 得到 0.597765/0.503268；由于同一 held fold 此前已经被查看，该
数字明确属于 selection-biased 互补性诊断，而不是可晋升 OOF。

INT4 对这些组件过于保守。骨干权重采用按输出通道 int8，输入 patch、相对位置偏置、
分类头、norm 和 bias 保持 fp16 后，两个传感器模型合计序列化为 57,007,698 bytes；
计入保留的 skeleton/temporal/detector 估算后为 67,657,705 bytes，尚未计最终 bundle
开销。因此下一次部署优先使用 int8，int6 仅作为大小门禁后备。该初筛阶段没有运行
匿名测试推理。在相同 batch-16 推理下，混合 int8 使 Depth 从 0.472067
变为 0.467877，IR 的整体 accuracy 保持 0.544693。

用户随后明确授权后，三个分支用全部 3,036 条样本、相同增强和固定等权 logits 规则
重新训练。无增强训练集 accuracy 为 fp32 0.950264、实际 mixed-int8 bundle 0.950593；
单文件 bundle 为 57,925,224 bytes。405 条测试预测自然覆盖 39 类（缺少类别 25），未
为补齐类别而修改任何预测。Kaggle submission 56035438 的 public score 为 0.58706，
与较弱 held-fold 证据一致且远低于 strictV3。该路线立即否决；排行榜结果没有用于调权
或再次提交。

### 5.14 strictV3 Temporal 主干加公共 Depth/IR，2026-09-05

下一项实验恢复 strictV3 Temporal 为主分支，并训练缺失的 B–E Depth/IR folds，得到完整
公共传感器 OOF。Depth、IR 分别为 0.492754、0.544466，Temporal 为 0.940053。唯一
冻结的概率规则 Temporal/Depth/IR=`0.90/0.05/0.05` 得到 0.939723；fold delta 为
−1/0/+1/0/−1，worst-user 保持 0.8125，因此未通过逐 fold 非退化门禁。

两个传感器共同修复 182 个 Temporal 错误中的 23 个，但在 Temporal 正确样本上共同
给出同一错误类别达 428 条。另一次严格 nested 权重诊断为 0.939065，并在 outer
B/C/D 选择传感器权重为零。融合接口予以保留，但 fail-closed sensor gate 固定为零，
即精确回退 Temporal；此后没有测试推理或 Kaggle 提交。

### 5.15 统一的 Omnivore Depth/IR 多层 Fusion，2026-09-05

为验证后期 logits 融合是否丢失跨模态结构，两个独立初始化的 Omnivore Swin-T trunk
输出全部四层特征（通道数 `192/384/768/768`）。空间池化后每层保留 8 个时间 token，
Depth 与 IR 在每一层通过双向四头 cross-attention 交换上下文，四层摘要最终进入同一个
Fusion 分类器；末层单模态头仅用于辅助监督。

固定 Fold-A 初筛采用 batch size 16、seed 2026 和 15 epochs。增强训练 accuracy 达到
0.816810，但 held accuracy 仅 0.512570，worst-user 为 0.431373；Depth、IR 辅助头
分别为 0.407821、0.502793。统一 Fusion 虽优于两个辅助头，仍低于独立 IR
（0.544693）、整段 token fusion（0.539106）和原 strictV3 Fusion 分支（完整 OOF
0.800395）。这是跨用户泛化失败；固定门禁据此停止 B–E 训练、匿名测试推理和 Kaggle
提交。

### 5.16 单 trunk Omnivore 原生 RGB-D token 融合，2026-09-05

单个 Omnivore Swin-T 随后接收 `[IR, IR, IR, inverse-depth]`，明确启用其四通道
`summed_rgb_d_tokens` 实现。appearance 与 Depth 分别 patch embedding，并在共享
transformer 前相加；均值和标准差仅由 2,320 条 outer-train 样本拟合，初始化只使用
官方公共 checkpoint。

固定 batch size 16、15 epochs 的 Fold-A 训练最终增强 accuracy 为 0.662069，held
accuracy 为 0.379888，worst-user 为 0.290640。虽然已直接验证原生代码路径确实启用，
结果仍低于独立 IR（0.544693）和双 trunk 多层模型（0.512570）。最可能的原因是预训练
契约不匹配：Omnivore 学习的是自然 RGB 加米制 Depth，本数据却是重复 IR 加 JET 反解
伪深度。门禁据此停止 B–E 训练、匿名测试推理和 Kaggle 提交。

### 5.17 strictV3 Temporal + Visual 严格等权融合，2026-09-05

该消融删除 legacy Fusion 并关闭逐样本 quality gate，在冻结的逐 fold/full temperature
之后给 Temporal、Visual logits 严格 `0.50/0.50` 权重。没有重新训练模型；未改变的
strictV3 bundle 为 69,805,793 bytes，包含全部推理权重且低于 100 MB。

Subject-wise OOF 从 strictV3 的 2,903/3,036（0.956192）降至 2,881/3,036
（0.948946）；A–E fold accuracy 为 0.959497/0.936107/0.979381/0.955010/0.901879。
405 条测试预测自然覆盖全部 40 类，相对 strictV3 改变 15 条。经用户明确授权的 Kaggle
提交得到 0.95522（ref 56036875），低于 strictV3 的 0.97512。该消融否决，发布契约
保持不变。

### 5.18 Temporal 主导的 Visual 优化，2026-09-05

成功的后续只做一个结构变化：将 legacy Fusion 的基础权重置零。strictV3 已冻结的逐
fold/full temperature、Temporal/Visual 权重和置信度 quality gate 全部继承。另一次
leave-one-fold-out 权重搜索仅得到 2,905 条，并使 A/E folds 退化，因此否决，没有用于
调整本候选。

继承式门控候选将 subject-wise OOF 从 2,903 提高到 2,909/3,036（0.958169）。Fold
delta 为 0/0/+5/+1/0，macro recall 从 0.953136 升至 0.956076，subject-macro 从
0.955794 升至 0.957954，worst-user 保持 0.8125。置信度门控使 Temporal 继续占主导，
测试集 Visual 平均有效权重为 0.1653；预测覆盖全部 40 类，仅改变 strictV3 的 3/405 条。

经授权提交后与标准 public score 0.97512 持平（ref 56036959）。该方案作为更简洁的
private-LB 候选保留，但公开榜持平不能证明它更优，也不会据此继续搜索权重。提交复用
合规的 69,805,793-byte bundle；裁剪已不使用的 Fusion 权重仍属于后续发布打包任务。

后续部署审计对 2,909 条 OOF 作出限定：该诊断没有应用仅存在于发布包的
`visual_package_output_scale=0.5`，而已提交 compact-package 推理实际应用了它。
0.97512 的提交结果仍然有效，但 2,909 不是部署一致的 OOF 估计。

### 5.19 Visual 架构优先级顺序实验，2026-09-07

三项 Visual 改动按顺序累加训练，均从同一公开 IG-65M → Kinetics-400 R(2+1)D-34
初始化开始，不加载任何项目 checkpoint。协议使用只由训练数据拟合的 Fold-A 预处理
契约、2,320 条训练数据、716 条 held-subject 数据、固定 15 轮，并只在终点评估一次。

control 为 448/716（0.625698）。零初始化的 layer2/3/4 时序残差 head 降至
436/716（0.608939）；保留 layer4 时间分辨率后找回 7 条，达到 443/716
（0.618715），但仍比 control 少 5 条；再加入 114 参数、零初始化、有界的 Depth/IR
门控后为 442/716（0.617318）。control 的最差用户 accuracy 为 0.517241，三个修改版
均为 0.497537。高时间分辨率结果说明末端时间降采样可能值得独立研究，但本次累加候选
均未通过 Fold A，因此停止 B–E、full-fit、匿名测试推理和 Kaggle 提交。

### 5.20 删除 Fusion 后的量化预算，2026-09-07

物理删除 Fusion 与未启用 thermal 占位权重后，单 checkpoint 降至 50,730,599 bytes，
T/V temperature、权重、quality gate 和 Visual output scale 均保持不变；405 条预测与
Fusion 权重置零候选完全一致。从原始 FP16 fold 权重重新量化后，主 Visual 没有收益：
5/6/8-bit 的最终 T+V 正确数分别为 2,893/2,892/2,893。可追溯 NTU120 代理从 FP16
1,788 降至 INT4 1,730，而 uniform 5-bit 达到 1,804；只保护末层的 mixed precision
仅 1,716，说明 INT4 误差分布在整个 trunk。附件中的另一份 1,998/1,859 源 checkpoint
不可用，因此没有用 INT4 伪造恢复。

在精确部署的 Visual 0.5 缩放下，重新生成的 5-bit T+V OOF 为 2,893/3,036；量化比较
应以它为锚点，而不是历史上未缩放的 2,909 条诊断。

### 5.21 Temporal 条件 Visual 纠错器，2026-09-08

已实现 `z_new = z0 + r(hV, zT)` 小型 head。隐藏投影正常初始化，仅最后一层为零，
所以初始输出严格等于 `z0`。固定目标为最终 CE，并在训练集中基线预测正确且置信度不低于
0.8 的样本上加入 `KL(p0 || p_new)`。

严格评估使用冻结的外部-only 编码器（PKU-MMD R(2+1)D-34 Visual 与 NTU DSTFormer
骨骼 Temporal）、全新固定终轮 40 类 head、20 组 outer×inner cross-fit，以及 5 个独立
outer-train head。每个目标上游 head 都排除 outer-held 用户，全程未访问匿名测试输入。
3,036 条基线从 1,201（0.395586）提高到 1,465（0.482543）：纠正 438、破坏 174，
净纠错 +264；A–E 分别为 +46/+58/+48/+66/+46，18 个用户全部改善。subject-macro
从 0.391962 提高到 0.479934，worst-user 从 0.1625 提高到 0.29375，按用户重采样的
accuracy delta 95% 区间为 [+0.07310,+0.10375]。第二次运行的两份 OOF 数组逐字节一致。

这证明条件纠错机制在严格隔离下有效，但不能估计部署 strictV3 的增益。外部-only 代理
显著更弱，其中 Visual 为 1,310 条、Temporal 仅 902 条，导致冻结的 strictV3 Temporal
主导融合权重失配。候选仍比 Visual 单支多 155 条，但该结果不授权 full-fit、匿名测试
推理或 Kaggle 提交。

随后进行了部署一致验证：从每折原始 FP16 权重重新做 uniform 5-bit 量化，严格采用
发布版单视图预处理与 0.5 输出缩放。得到的 T+V 基线为 2,890/3,036（0.951910），
测试预测与已提交的 T+V CSV 逐行一致。使用分类头前 512 维特征的纠错器净下降 24 条；
唯一预注册的坐标对齐改动（改用 40 维类别 logits）仍净下降 11 条：纠正 32、破坏 43，
A–E 折为 +6/+9/+6/0/-32。subject-macro 从 0.951297 降至 0.947341，worst-user
从 0.8125 降至 0.625。

误差呈现明确的泛化偏置：用户 5 中 30 条原本正确的类别 36 被改判为类别 10，解释了
绝大部分 E 折失败。两个预注册部署筛选均未通过非退化门禁，因此未 full-fit，也未提交
Kaggle；否则会把已失败的确认性终点转化为 leaderboard 引导调参。

### 5.22 输入有效性与动作不变性对照，2026-09-08

基于源文件实际解码、而非黑像素启发式的审计发现：2,933 条样本有可用 Depth 或 IR，
2,931 条至少有一帧通过骨架解析；103 条两个主输入都无效。冻结发布模型在这 103 条上
仅正确 30 条，因此它们贡献了 133 个 OOF 错误中的 73 个；其中用户 5 占 60 条、类别
36 占 30 条。这确认了严重的缺失混杂，以及它与纠错器类别—用户失败位置的重合。

随后在不访问匿名测试集的前提下，完成三项预注册、固定终轮的 subject-OOF 对照。第一，
所有样本仍参与前向和 BatchNorm，只对分支无效行屏蔽 CE 梯度。冻结门控 T+V control
为 2,889；仅屏蔽 Visual、仅屏蔽 Temporal、同时屏蔽分别为 2,888/2,885/2,885。
同时屏蔽纠正 8、破坏 12、净 -4，A–E 为 -1/-5/+2/0/0；有效输入子集与
subject-macro 也下降。

第二，固定绝对骨长的初版构造在训练前被几何质检拒绝，最大归一化位移达到 2.474。
通过质检的版本使用 clip 固定、左右协调的 ±5% 骨长倍率，保留投影运动、方向、root、
置信度与缺失帧，平均/最大位移为 0.0153/0.1460。Temporal 为 2,847→2,846，T+V 为
2,889→2,888，没有纠正样本、破坏 1 条。

第三，在 TCN 分类器实际使用的 40 维表示上加入权重 0.05、温度 0.1 的监督对比项；
正样本为同类不同 outer-train 用户，不同类为负样本，同类同用户被忽略。每折每个 epoch
至少有 2,118 个有效 anchor，且对比损失持续下降，但 Temporal 为 2,847→2,843，T+V
为 2,889→2,888（纠正 5、破坏 6；A–E 为 -1/-1/+1/0/0）。因此问题并非没有形成
训练信号。三个候选均未通过门禁，结果揭示后没有继续扫描 mask、幅度、权重或温度。

### 5.23 在 40 类单帧分类头之前进行时序建模，2026-09-09

重训前先按源输入有效性分解 §5.22 的三项处理。它们在 103 条双主输入均无效的样本上
均为 0 纠正、0 破坏；所有变化都发生在 2,931 条双主输入有效样本中。这说明负结果确实
来自动作判别变化，而不是改变了缺失输入的默认输出。

随后用预注册配对实验隔离检验：最终 40 类单帧分类头是否丢失运动证据。同一次发布
DSTFormer 前向同时导出 frame logits 与 `fc2` 前、ReLU 后的 2,048 维表示；logits 与
既有对照缓存最大差异为 0，SHA256 完全一致。每个外层折的 PCA-40 只在 outer-training
用户的帧上拟合并冻结，解释方差总和为 0.655–0.670。静态路径始终是同一组 16 帧 logits
的均值，唯一变化是同容量 TCN 的输入由 logits 改为 PCA 特征；结构、随机种子、训练
预算、Visual 输出和发布 T+V gate 均不变。

Feature-TCN 使 Temporal 从 2,847 降到 2,839；经冻结 T+V 融合后从 2,889 增到
2,891，即纠正 6 条、破坏 4 条，各折净变化为 0/-1/+2/0/+1。两条净增益全部发生在
主输入有效样本上，103 条缺失样本不变。用户宏平均增加 0.000777，形式上通过预注册的
最小门槛；但双侧精确 McNemar p=0.7539，按用户聚类 bootstrap 的准确率变化 95% 区间
为 [-0.000983, 0.002628]，而 Temporal 单支在三折下降。因此这只是与 Visual 融合交互
产生的弱证据，不能确认 40 维分类头是主要时序瓶颈；同一用户和易混类别内部也同时出现
纠正与破坏。该候选不晋级、不 full-fit、不提交，也不继续扫描 PCA 维数或 attention。
局部视图实验仍必须以原始帧直接显示裁剪、分辨率或 16 帧采样损失为前提。

### 5.24 当前 T+V 错误覆盖与 raw-to-cache 证据，2026-09-09

错误名单已重置为当前配对 T+V control，而非沿用旧 2,903 条正确的发布结果。当前
2,889/3,036 OOF 共 147 个错误：103 条双主输入缺失样本中占 73 个，此外还有 74 个
（72 个双主输入有效、2 个只有一支有效）。这 74 个中，Temporal 独有 top-1 正确 6 条，
Visual 独有正确 53 条，两支 top-1 都错仅 15 条。后 15 条的真实类别在 Temporal 中
全部排名第 2；Visual 中 11 条第 2、1 条第 3，另外三条分别第 13/25/36。因此当前有效
输入错误主要涉及已有分支证据没有被最终组合保留。这既不是软融合上限，也不授权根据
这些 OOF 错误调整分支权重。

随后只把六个直接可比的已完成候选用于确定审阅优先级：它们合计曾纠正 10 个不重复的
基线错误，137 个持续未纠正。持续集合包含全部 73 个双主输入缺失错误、6 个 Temporal
独有正确、45 个 Visual 独有正确和 13 个两支 top-1 都错。没有拼接候选输出，也没有把
这些行转成训练目标。

输入证据盲审从十个独立持续双错簇各取一条，并匹配同类、不同用户的正确对照。打开
标签、预测和错误/对照身份前，逐条比较全部 Depth_Color 原始帧、实际选择的 16 张未裁剪
帧、发布契约的 union crop 后 128×128 Depth/IR，以及对应 H36M-17 骨架。两组均未出现
明确的采样阶段或空间处理损失；骨架可疑异常在错误和对照中均为 2/10，20 条样本的
Depth、IR、Skeleton 源帧数全部一致。骨架最大归一化帧间跳变中位数在错误组为 0.488、
对照组为 0.629。该小规模描述样本不能证明原始数据没有可恢复信息，但没有揭示可授权
采样、局部视图、分辨率或骨架修复训练的跨用户重复机制。

---

## 6. 通用方法纪律（硬约束）

1. **绝不**使用测试/匿名标签、预测历史、提交分数或排行榜逐行反馈作为设计信号。任何接触这些信息的候选立即作废。
2. **Outer A–E / inner C–E 是确认性终点**，不是反馈通道。一个候选一旦揭示这些结果，就不得在同一方向继续扫描超参数。
3. 每个新候选都必须附带新的 preregistration、新 formal route、新独立审计，并关闭 P0/P1。不接受“已有候选只改一行”。
4. **强制门禁顺序**：
   * full-data OOF nested-CV：5/5 fold 非退化、subject-macro 同方向、worst-user 改善且跨 seed 稳定；
   * 两轮原始数据重放：CSV 逐字节一致、logits `max_abs` ≈ 0，并保证 deployment 与 training OOF 的 bit-width / precision 匹配；
   * 自然覆盖 40 类；
   * 一个 checkpoint 包含全部推理权重，且 < 100,000,000 bytes；
   * 独立审计 SHA 一致；
   * 只有全部通过后才能生成 `submit_candidate.zsh`，且仅在 Kaggle quota 允许时提交。
5. **禁止放宽标准**：独立审计发现真实 P0/P1 后，候选立即作废。不得为了接纳候选而放宽 package 预算、门禁或审计要求。

---

## 7. 下一步具体工作清单（按顺序）

1. 将已完成逐字节重放的 `sched30_consensus` 作为冻结的负面部署证据保留；其公开分数
   为 0.97014，不得调参后重提。
2. 为从零训练的 TSM/S3D 路线完成 outer-train-only normalization builder 与 matched CV runner 的 **CPU materialisation**；只读取有标签训练 cache 与 metadata，绝不打开 held/test/anonymous/submission。运行 80 个 inner 与 30 个 outer synthetic regression。
3. **关闭 PKU/NTU 外部 IR。** 保留已完成的 depth-only PKU/NTU 证据，不训练或汇报
   不完整配对 IR；§5.8 公开榜持平后也继续关闭 PKU 部署路线，不根据排行榜重新调整
   bridge 权重、precision、epoch、seed 或唯一变化的样本。
4. 任何新机制候选均须同步记录到 `BEST_REPORT_EVIDENCE_MANIFEST_*.json`（包含 SHA 和决策）以及 `EXTERNAL_ONLY_RESEARCH_ROADMAP_20260830.md`（将原方向标记为 authorized 或 rejected）。
5. 报告与证据收尾：上述每个已执行步骤都必须将结果写入对应报告的 fact-freeze-date 段落，并附上新的 manifest SHA。

---

## 8. 复现与审计 strictV3

artifact 级检查会复现报告中的 OOF 分数和标准 CSV。原始数据重放会逐字节复现计分为 0.97512 的 CSV：

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --output results/strict_v3/reproduced_submission.csv
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify \
  --replayed-csv results/strict_v3/reproduced_submission.csv
```

任何不满足标准 hash 的重放都会 fail closed。

## 9. 合规与规则摘要

本项目从不使用测试/匿名标签、ID、样本、提交分数或事后排行榜反馈作为设计信号。9 月的重放提交仅用于确认性复现审计，其分数未被用于调整两条差异预测。§5.8 的外部数据诊断同样在匿名推理前完全冻结；持平分数只被记录，随后关闭路线而不继续调参。历史合规链已汇总在第 4.6 节。

## 10. 证据锚点（仓库内路径）

* [`checkpoints/strict_v3/model.pt`](../checkpoints/strict_v3/model.pt) 和 [`yolo11n.pt`](../checkpoints/strict_v3/yolo11n.pt)——可部署模型。
* [`results/strict_v3/release_manifest.json`](../results/strict_v3/release_manifest.json) 和 [`metrics.json`](../results/strict_v3/metrics.json)——发布契约与 OOF 证据。
* [`results/strict_v3/submission.csv`](../results/strict_v3/submission.csv)——公开榜分数为 0.97512 的标准 CSV。
* [`results/strict_v3/raw_replay/`](../results/strict_v3/raw_replay/)——逐字节一致的原始数据重放。
* [`verify.py`](../yolo_r2plus1d/strict_v3/release/verify.py) 和 [`replay.py`](../yolo_r2plus1d/strict_v3/release/replay.py)——公开验证与端到端重放入口。
