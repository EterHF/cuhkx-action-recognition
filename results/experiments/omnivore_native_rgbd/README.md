# Native Omnivore RGB-D token fusion

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

One official Omnivore Swin-T trunk receives aligned
`[IR, IR, IR, inverse-depth]` clips. Four-channel input activates Omnivore's
native `summed_rgb_d_tokens` path: separate appearance and depth patch
embeddings are summed before the shared transformer. Sensor normalization is
fit only on the 2,320 Fold-A training rows. No project checkpoint is loaded.

The fixed 15-epoch, batch-16 run reached 0.662069 augmented training accuracy.
On 716 held-subject rows, accuracy was 0.379888 and worst-user accuracy was
0.290640. This is below independent IR (0.544693), dual-trunk multi-stage
fusion (0.512570), and strictV3 Visual (0.896904 full OOF).

The native path works mechanically, but its pretraining contract assumes
natural RGB plus metric depth. Replacing RGB with repeated IR and metric depth
with JET-inverted pseudo-depth creates an early-fusion domain mismatch. The
Fold-A gate stopped B–E training, anonymous-test inference and Kaggle submission.
