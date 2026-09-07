# Fusion-free deployment and quantization budget

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

The release package now has a tested construction path that physically removes
`fusion_4bit` and the inactive thermal placeholder. The retained Temporal and
Visual temperatures, base weights, quality gate, and
`visual_package_output_scale=0.5` are unchanged. The resulting model is
45,111,978 bytes; with the embedded YOLO detector, the single checkpoint is
50,730,599 bytes and has 49,269,401 bytes of margin. Its 405 predictions exactly
match the previously submitted zero-Fusion-weight candidate.

Quantization was then repeated from the original FP16 weights, never from a
dequantized low-bit package. On the 3,036-row subject-wise OOF, the current
Visual branch and final T+V result were:

| Policy | Visual correct | T+V correct | Bundle bytes |
| --- | ---: | ---: | ---: |
| FP16 reference | 2,754 | 2,893 | over budget / not built |
| uniform 5-bit | **2,755** | **2,893** | 50,732,895 |
| uniform 6-bit | 2,747 | 2,892 | 58,666,015 |
| uniform 8-bit | 2,754 | 2,893 | 74,532,191 |
| 5-bit + layer4/head 8-bit | 2,754 | 2,893 | 65,385,317 |
| 6-bit + layer4/head 8-bit | 2,746 | 2,892 | 68,437,719 |

Higher precision fits comfortably but produces no positive final net
correction, so the main Visual remains 5-bit.

The deployment-matched 2,893 baseline is intentionally lower than the earlier
2,909 inherited-gate diagnostic. That diagnostic fused the frozen Visual OOF
artifact without the package-only 0.5 output scale, while test inference used
the compact package and did apply the scale. The 0.97512 Kaggle observation
remains factual, but 2,909 is not a deployment-matched OOF result.

The exact attachment candidate with 1,998 FP16 and 1,859 INT4 correct rows has
no original float checkpoint in the workspace. A clearly labelled traceable
NTU120 proxy nevertheless confirms the mechanism: FP16 scored 1,788, INT4
1,730, and fresh uniform 5-bit 1,804. Raising only layer4/head to 8-bit scored
1,716, so INT4 damage is not concentrated only in late layers. BRECQ-style
activation reconstruction is deferred because simple 5-bit already closes the
proxy's FP16 accuracy gap; the exact attachment model still requires its
original FP16/FP32 checkpoint.
