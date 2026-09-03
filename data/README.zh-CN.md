# CUHK-X — 小模型赛道数据说明

[English](README.md) | [简体中文](README.zh-CN.md) | [仓库首页](../README.zh-CN.md)

这是一个多模态**人体动作识别（分类）**任务。给定一个多模态 clip，预测其动作类别（`action_id`，范围 **0–39，共 40 类**）。

## 目录结构

```text
.
├── Training/
│   ├── class_mapping.csv     # action_id <-> action_name（40 类）
│   └── data/
│       └── HAR.z01 … HAR.z08 + HAR.zip   # 分卷 zip
│           →  HAR/data/<modality>/<action>/<user>/<trial>/<files>
└── Testing/
    ├── data/
    │   └── small_model_track_test.zip    →  small_model_track_test/<id>/<modality>/<files>
    └── test_file/
        ├── test.csv              # path + 待填写的空 `prediction`
        └── sample_submission.csv # 提交示例
```

## 标签

- **训练标签位于路径中**：在 `HAR/data/<modality>/<action>/<user>/<trial>` 中，`<action>`（例如 `0_Wash_face`）就是类别。
- 使用 `class_mapping.csv` 转换 `action_name` 与 `action_id`。
- 测试 clip 已匿名化为 `SM_test_XXXX`，需要预测对应的 `action_id`。

## 模态（共 6 种；没有 RGB，也没有原始 Depth）

| 模态 | 类型 | 文件示例 |
| --- | --- | --- |
| `Depth_Color` | 彩色化深度帧 | `Depth_<datetime>_<idx>_Color.png` |
| `IR` | 红外帧 | `IR_<datetime>_<idx>.png` |
| `Thermal` | 热成像帧 | `frame_000063.jpg` |
| `IMU` | 惯性传感器 | `*.csv` |
| `Radar` | 毫米波雷达 | `radar_output_T<ts>.csv` |
| `Skeleton` | 骨架 | 姿态数据及 `visualizations/` |

各模态采样率不同，并非每个 clip 都包含全部模态。

## 解压数据

训练集是分卷 zip（`HAR.z01`…`HAR.z08` 与 `HAR.zip`，须放在同一目录）：

```bash
cd Training/data
zip -s 0 HAR.zip --out HAR_full.zip   # 合并分卷（需要 zip 3.0+）
unzip HAR_full.zip                    # -> HAR/data/<modality>/<action>/<user>/<trial>/...
```

也可使用 7-Zip、WinRAR 或支持分卷压缩包的图形界面工具。测试集：

```bash
cd Testing/data && unzip small_model_track_test.zip   # -> small_model_track_test/<id>/<modality>/...
```

## 提交格式

在 `Testing/test_file/test.csv` 中，将每行的 `prediction` 填为预测的 `action_id`（0–39）。格式参见 `sample_submission.csv`。

## 数据统计

- 40 个动作类别
- 405 个测试 clip

## 快速开始

```python
import csv
id2name = {r["action_id"]: r["action_name"]
           for r in csv.DictReader(open("Training/class_mapping.csv", encoding="utf-8-sig"))}
# 训练数据路径中的 <action> 文件夹即标签，例如：
#   HAR/data/IR/0_Wash_face/user10/4-2-1/
#   -> action_name="0_Wash_face", action_id="0"
```
