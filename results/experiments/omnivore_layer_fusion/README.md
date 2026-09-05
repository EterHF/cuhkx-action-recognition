# Multi-stage Omnivore Depth/IR fusion

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

Two Omnivore Swin-T trunks were initialized only from the official public
checkpoint. Their four intermediate stages (`192/384/768/768` channels) retain
eight temporal tokens after spatial pooling. At every stage, Depth and IR use
bidirectional four-head cross-attention; the four fused summaries feed one
classifier. Final-stage auxiliary heads keep each modality supervised.

The fixed Fold-A screen used 2,320 training rows, 716 held rows, batch size 16,
seed 2026 and 15 epochs. Training accuracy reached 0.816810. Held accuracy was only 0.512570
with worst-user 0.431373. Held auxiliary accuracies were 0.407821 for Depth and
0.502793 for IR. The fused head learned some cross-modal signal, but remained
below independently trained IR (0.544693), clip-token fusion (0.539106), and
the original strictV3 Fusion branch (0.800395 full OOF).

The failure is cross-user generalisation, not convergence or missing layer
access. The B–E expansion, test inference and Kaggle submission were stopped by
the preregistered Fold-A gate.
