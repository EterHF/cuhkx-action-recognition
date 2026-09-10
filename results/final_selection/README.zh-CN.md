# 最终提交候选

[English](README.md) | [简体中文](README.zh-CN.md) | [项目状态](../../docs/PROJECT_STATUS.zh-CN.md)

**网站勾选尚未完成或核实。** 当前 Kaggle OAuth 登录可通过普通 API 读取、下载提交，
但网站最终选择接口返回 `401 Unauthenticated`。没有把推荐结果冒充为已选中。
需在已登录的[比赛提交页](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/submissions)
勾选下列两条，并确认最终恰有这两个编号。

| 提交编号 | 方案 | 提交时间（UTC） | 文件名 | 公开分数 | 全部权重 |
| --- | --- | --- | --- | ---: | ---: |
| **55712568** | strictV3 原发布版本 | 2026-08-23 10:35:47 | submission1.csv | **0.97512** | 69,805,793 B |
| **56036959** | Temporal + Visual 继承门控 | 2026-09-05 16:34:50 | submission.csv | **0.97512** | 50,733,469 B |

两份提交都已从 Kaggle 下载，逐字节匹配仓库中的 CSV，且相差 **3/405** 行。
第二条不拟合新参数，只删除弱 Fusion 分支；独立包也移除了不参与推理的权重。
选择依据是可复现、同等已知公开表现和结构差异，不能保证 private LB 更优。
早期 T+V 的 2,909 OOF 存在 Visual scale 不一致，未被用作本次选择理由。
未优先选择的新 selector 为 0.96019；gate-off 虽公开持平，但没有通过其原 OOF 门槛。

## 文件与核验

* 原发布 CSV：[`results/strict_v3/submission.csv`](../strict_v3/submission.csv)
* T+V CSV：[`results/experiments/temporal_visual_inherited/submission.csv`](../experiments/temporal_visual_inherited/submission.csv)
* 原发布包：[`submission_bundle.pt`](../../checkpoints/strict_v3/submission_bundle.pt)
* T+V 独立包：[`temporal_visual_inherited_bundle.pt`](../../checkpoints/experiments/temporal_visual_inherited_bundle.pt)
* 机器可读记录：[`selection.json`](selection.json)，包括哈希、两次原始重放与网站 401 状态。

两个包分别满足全部权重单文件严格小于 100,000,000 bytes 的要求。
T+V 在两次独立原始重放中 CSV 逐字节一致、分支 logits 最大差值为零，并与 Kaggle 下载
的历史 `56036959` 文件一致；没有为最终选择再次提交或修改任何预测。

从仓库根目录重放第二条：

```bash
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --bundle checkpoints/experiments/temporal_visual_inherited_bundle.pt \
  --output runs/final_selection_replay/submission.csv
cmp runs/final_selection_replay/submission.csv \
  results/experiments/temporal_visual_inherited/submission.csv
```

原发布版本使用首页的 `release.verify` 与原始重放命令。
