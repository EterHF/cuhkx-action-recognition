# strictV3 Temporal + Visual equal fusion

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

This ablation removes the legacy Fusion branch and combines the existing
strictV3 Temporal and Visual logits with fixed `0.50/0.50` weights after their
frozen fold/full temperatures. The per-sample quality gate is disabled, making
the deployed weights exactly equal. No model was retrained and the unchanged
69,805,793-byte strictV3 bundle remains below the official 100 MB limit.

Five-fold subject OOF was 2,881/3,036 (`0.948946`), below release strictV3's
2,903/3,036 (`0.956192`). Fold accuracies were A–E `0.959497/0.936107/`
`0.979381/0.955010/0.901879`. Test predictions naturally cover all 40 classes
and differ from strictV3 on 15 of 405 rows.

Kaggle submission ref `56036875` scored `0.95522`, below strictV3's `0.97512`.
The ablation is rejected and the score is not used to tune another mixture.
