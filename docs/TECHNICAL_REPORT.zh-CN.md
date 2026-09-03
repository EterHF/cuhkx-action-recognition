# CUHK-X 小模型赛道 — 技术报告

[English](TECHNICAL_REPORT.md) | [简体中文](TECHNICAL_REPORT.zh-CN.md) | [文档索引](README.zh-CN.md) | [仓库首页](../README.zh-CN.md)

**目前最佳 Kaggle 公开榜分数：`0.97512`（第 3 名，ref `55712568`）**<br>
**与第 2 名的差距：`0.00497`（相当于一个公开样本的净变化）。**<br>
**冻结日期：2026-09-03**

> 本报告将历史改进路线和后续策略合并为一份文档。它是项目唯一的顶层技术叙事；逐候选提交日志、strict-v3 优先级备忘录、清理 manifest，以及来自 Codex 任务 `01a04b41…` 的后续方向均汇总于此，不再作为独立文件发布。

---

## 1. 问题与数据

CUHK-X 小模型赛道（[挑战页面](https://openaiotlab.github.io/CUHK-X-Challenge/)，UbiComp / ISWC 2026）是一个包含 40 个类别的跨用户动作识别任务，每个 clip 包含六种模态：

- 8 帧有序的 **Depth（Color / IR）** 与 **Thermal** 图像，采用保持宽高比的 padding。
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
| Package | `checkpoints/strict_v3/model.pt`（64,206,000 bytes） |
| 标准提交 | `results/strict_v3/submission.csv`（405 行、40 类） |
| 原始数据重放提交 | `results/strict_v3/raw_replay/submission.csv`（405 行、40 类） |
| 连同 YOLO11n 总大小 | 69,819,764 bytes（低于 100 MB） |
| 公开榜分数 | **0.97512**，ref `55712568` |
| 原始数据重放分数 | **0.97512**，与 ref `55712568` 逐字节一致 |
| OOF（release-strict） | aggregate `0.956192`，mean fold `0.954823`，worst fold `0.926931`，macro recall `0.953143` |
| 协议 | `five_fold_subject_wise_nested_temperature_quality_gate`，完整契约见 `release_manifest.json` |
| 原始重放证据 | 确定性 SHA-256 `e2509491…`，与标准 CSV 逐字节一致 |

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

多个离线“更高 OOF”候选（如 OOF 为 0.970* 的 Fusion4 raw）均被公开排行榜否定，不再属于候选方案。

### 3.2 复现 `legal_strict_v3`

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --output results/strict_v3/reproduced_submission.csv
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify \
  --replayed-csv results/strict_v3/reproduced_submission.csv
```

验证器要求原始数据重放结果与标准 CSV 逐字节一致。配方记录在 `release_manifest.json`；推理契约的任何变化都必须生成新的 package 和 manifest。

### 3.3 清理后仓库的重训练审计

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

本节是完整审计链。每个方向均列出结果、停止原因和标准证据文件。多数方向已经“穷尽”，继续投入只会扩大验证集搜索空间。

### 4.1 原始 EfficientNet-B0 基线（历史）

EfficientNet-B0 多模态基线、辅助损失选择 fold（`pretrained_aux015`，mean OOF ≈ 50.93%）以及三随机种子 ensemble（`pretrained_final_seed{2026,3407,8819}`、`pretrained_ensemble/`）构成了最初的 **release-strict** anchor。这些权重和脚本未随清理后的 main 分支发布，测得结果保留在此以记录来源。

### 4.2 视觉 / 时序 / 融合分支开发（保留为证据）

YOLO crop 的 R(2+1)D-18 → R(2+1)D-34 int5/int6 + YOLO11n 路线将纯视觉候选从 0.46268 提升至 0.71641。加入 ST-GCN/DSTFormer 风格 Skeleton 融合与时序 TCN residual head 后，得到合法的“5-bit multibranch”package。strict-v3 发布版本对这些分支应用按 fold 的 nested selection 规则，不使用排行榜反馈，也不泄漏样本或用户 ID。

### 4.3 外部数据扩展（已穷尽）

项目对 NTU Depth / Skeleton 预训练与 PKU-MMD bridge 做了大量探索。审计汇总如下：

| 实验 | 结果 | 结论 |
| --- | ---: | --- |
| NTU masked-depth 预训练 + R(2+1)D-18 | 62.98%（control 58.27%），5/5 fold 提升，worst-user +12.50 pp | 小 backbone 上存在真实正迁移信号 |
| NTU60 / NTU120 直接监督 R(2+1)D-34 | 58.56% / 58.89% | 大规模源域监督训练会**遗忘** Kinetics 表示 |
| Kinetics anchor + 25% NTU60 encoder 插值 | 65.18% | 应保留 Kinetics anchor，但未改善最终 strict-v3 |
| NTU Skeleton student | source val 在 epoch 25 达到 64.68% | 外部 Skeleton encoder 可用，但仍需五折目标域验证 |
| PKU-MMD source-only 预训练 | source val 75.30%，source train 在 epoch 15 约 99% | 6,952 条记录明显过拟合；源域验证无法代理目标域迁移 |
| Kinetics → PKU → CUHK-X | 61.92%（control 63.27%） | 直接 PKU continued-training 是**负迁移** |
| 用 PKU visual 替换 strict-v3 visual | 95.191%（control 95.619%）；nested mean weight = 0 | 不能作为第 4 个 logit 分支 |

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
* **PKU-MMD 许可证**：公开页面没有授予奖金竞赛中的再分发权；在获得书面许可或澄清前，V3 GPU handoff **保持暂停**。
* **MViTv2-S**：官方 checkpoint 约 131.9 MB，超过 100 MB 限制。**暂时搁置**，等待组织方书面分类意见。
* **TorchVision 许可证提醒**：预训练权重可能继承其训练数据条款，已记录在 External-Compliance 审查中。

---

## 5. 仍在考虑的方法与实验

这些方向构成归档研究队列。它们记录在此，以确保清理后仍能保留设计依据；其 preregistration、实验和协议代码有意不随 strictV3-only main 分支发布。

### 5.1 候选融合的严格 nested selection（优先级 1）

* **动机**：所有手工选择的系数和由排行榜驱动的单行修改都已被否定。唯一可接受的选择规则是：“每个 held-fold 的系数只能由其余四个 fold 的 OOF 决定，不得使用用户/样本 ID 或测试集属性”。
* **已实现（CPU / synthetic）**：Fusion7 nested selection（OOF 净增 14 行、5/5 非退化、自然覆盖 40 类）；50/50 probability-mean 对 strict-v3（5/5 非退化、净增 9 行、距离接收标准差 1 行）；nested coefficient grid（端点及 0.25 / 0.5 / 0.75）；shared-state multi-pooling 候选（energy + top-2，共用 Fusion4 / Visual4 / DSTFormer，OOF 净增 12 行、5/5 非退化、自然覆盖 40 类，大小为 86.43 MB，连同 YOLO 为 92.05 MB，远低于 100 MB）。
* **待完成步骤**：deployment / training-OOF 的精度与 bit-width 匹配；已知存在 FP16 ↔ FP32 DSTFormer 差异。
* **后续设计要求**：
  1. selector 只能读取其余四个 fold 的 OOF logits、marginal 和 agreement 信号。禁止使用用户或样本级属性。
  2. 选择规则必须预先声明，不允许“查看 fold C 后再调整 threshold”。
  3. 只有在两轮原始数据重放一致且完整 nested gate 通过后，候选才能加入提交队列。

### 5.2 等权概率 consensus / multi-pooling（优先级 2）

* **动机**：量化、校准和 TTA 路径均已穷尽后，“相同权重、不同 pooling”仍可修正少量错误且不增加新权重，能够维持 100 MB 预算。
* **已实现**：
  * `sched30` 三随机种子时序 consensus（FP32）：OOF +0.004282、5/5 非退化、每个 seed 均为 5/5；大小 92.05 MB，连同 YOLO 仍为 92.05 MB；CSV hash 为 `f33e0569…`。
  * 时序 pooling 多数投票（top-2 + energy）：OOF +0.003623、5/5 非退化；大小 88.43 MB，连同 YOLO 为 96.96 MB；CSV hash 为 `6a320486…`。
* **后续设计要求**：
  1. 将两个 pooling variant 视为结构上的最小集合，并执行严格 OOF nested selection，不能使用固定 50/50 融合，也不能手工挑选系数。
  2. 只有与 anchor / `sched30` 共享 DSTFormer frame logits 且不增加大型权重的新 pooling variant 才可接收。
  3. 每个候选必须生成两份逐字节相等的 CSV，并确保 deployment logits 与 training OOF 在 FP16/FP32 下精确匹配。

### 5.3 从零训练的小 backbone——“仅竞赛使用”（优先级 3，合规 fallback）

* **动机**：小模型赛道的 ≤100 MB 规则、“不得使用大型预训练 backbone”的要求、PKU-MMD 许可证歧义，以及 MViTv2-S 约 131.9 MB 的 checkpoint，共同要求一条完全从零训练的不同方法。
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

### 5.4 外部表示迁移（优先级 4——仅在 5.1–5.3 失败时）

外部数据研究队列保持如下：

1. **VideoMAE-S ≈ MViTv2-S**：“≈”表示相同调查优先级，而非二者在本数据集等价。在打开任何 CUHK-X fold 前冻结外部 checkpoint、架构、预处理和 seed map。
2. **确定性 depth / lag-1 temporal-difference channel**：配方固定、可审计，并预先声明辅助权重；不得根据 held 结果搜索 channel recipe。
3. **正确的 SWA**：预先声明平均区间、LR schedule、参数范围和 BN 处理方式。平均后的 checkpoint 是唯一候选，**绝不能**选择 held 上的最佳 epoch。
4. **SlowFast / X3D**：沿用相同 subject folds、seed budget 和 source-only 数据边界；先在本地测量计算量和准确率。
5. **MixStyle / ASAM**：排在最后，因为小 fold 上的偶然增益往往损害跨 seed 稳定性。

External-only 的 strict 定义是：外部表示、公开架构和确定性训练变换。**绝不能**使用测试/匿名标签、ID、样本、提交分数、预测历史或事后排行榜反馈。

---

## 6. 通用方法纪律（硬约束）

1. **绝不**使用测试/匿名标签、预测历史、提交分数或排行榜逐行反馈作为设计信号。任何接触这些信息的候选立即作废。
2. **Outer A–E / inner C–E 是确认性终点**，不是反馈通道。一个候选一旦揭示这些结果，就不得在同一方向继续扫描超参数。
3. 每个新候选都必须附带新的 preregistration、新 formal route、新独立审计，并关闭 P0/P1。不接受“已有候选只改一行”。
4. **强制门禁顺序**：
   * full-data OOF nested-CV：5/5 fold 非退化、subject-macro 同方向、worst-user 改善且跨 seed 稳定；
   * 两轮原始数据重放：CSV 逐字节一致、logits `max_abs` ≈ 0，并保证 deployment 与 training OOF 的 bit-width / precision 匹配；
   * 自然覆盖 40 类；
   * package + YOLO ≤ 100 MB；
   * 独立审计 SHA 一致；
   * 只有全部通过后才能生成 `submit_candidate.zsh`，且仅在 Kaggle quota 允许时提交。
5. **禁止放宽标准**：独立审计发现真实 P0/P1 后，候选立即作废。不得为了接纳候选而放宽 package 预算、门禁或审计要求。

---

## 7. 下一步具体工作清单（按顺序）

1. **完成 nested / shared-state multi-pooling 的两轮原始数据重放**，覆盖 `sched30` 和 `fp32_consensus`。门禁通过后将其提升至提交队列首位。
2. 为从零训练的 TSM/S3D 路线完成 outer-train-only normalization builder 与 matched CV runner 的 **CPU materialisation**；只读取有标签训练 cache 与 metadata，绝不打开 held/test/anonymous/submission。运行 80 个 inner 与 30 个 outer synthetic regression。
3. **只有 1 和 2 仍无法突破公开榜前二时**，才为优先级 4 的外部研究队列申请新的 preregistration。
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

本项目从不使用测试/匿名标签、ID、样本、提交分数或事后排行榜反馈作为设计信号。9 月的重放提交仅用于确认性复现审计，其分数未被用于调整两条差异预测。历史合规链已汇总在第 4.6 节。

## 10. 证据锚点（仓库内路径）

* [`checkpoints/strict_v3/model.pt`](../checkpoints/strict_v3/model.pt) 和 [`yolo11n.pt`](../checkpoints/strict_v3/yolo11n.pt)——可部署模型。
* [`results/strict_v3/release_manifest.json`](../results/strict_v3/release_manifest.json) 和 [`metrics.json`](../results/strict_v3/metrics.json)——发布契约与 OOF 证据。
* [`results/strict_v3/submission.csv`](../results/strict_v3/submission.csv)——公开榜分数为 0.97512 的标准 CSV。
* [`results/strict_v3/raw_replay/`](../results/strict_v3/raw_replay/)——逐字节一致的原始数据重放。
* [`verify.py`](../yolo_r2plus1d/strict_v3/release/verify.py) 和 [`replay.py`](../yolo_r2plus1d/strict_v3/release/replay.py)——公开验证与端到端重放入口。
