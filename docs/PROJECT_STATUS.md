# Project status and research priorities

[English](PROJECT_STATUS.md) | [简体中文](PROJECT_STATUS.zh-CN.md) | [Home](../README.md)

Updated 2026-09-10. The best public score remains **0.97512**. These experiments
do not establish a performance ceiling or guarantee reaching the top two. The
clearest limitation is that engineering OOF gains have not reliably translated
into public gains. Repeatedly screening small changes on the same OOF data has
limited value.

## Latest completed experiments

| Direction | Evidence | Decision |
| --- | --- | --- |
| Thermal TSN / TSM / person crops | 45 models; deployment-seed user-cluster intervals include zero | Not submitted |
| Local/world IMU | 30 models; world mean 30.87% versus local 31.41% | Not submitted |
| Available-input residual correction | 15 models; net +17/+20/+17, all lose Fold-E rows | Not submitted |
| Binary branch selection | First 15 models net +23/+25/+27, two seeds lose Fold-D rows | No standalone submission |
| Three-seed unanimity | 15 additional replication models; both triplets net +26, no degrading fold | Public 0.96019; rejected |

See the [experiment index](../results/experiments/README.md) for recipes and negative
evidence. New seeds are not independent data. Target-pretrained upstream models
and repeated use of held results make these OOF metrics engineering comparisons,
not unbiased estimates of private-LB performance.

## Remaining research opportunities

1. **Improve validation credibility first.** Freeze upstream weights and recipes
   before target adaptation, excluding outer-held users from the outset. Use
   genuinely nested inputs for inner selection. Previously inspected users cannot
   become a fresh confirmation set retroactively; prioritize independent users
   when new data is available.
2. **Improve representations or test teacher distillation.** A stronger public
   video/skeleton teacher used during training may improve the small model more
   than another logit correction. Evaluate the actual quantized deployment state.
   This remains an untested hypothesis here, not a promised score gain.
3. **Improve temporal evidence for missing primary inputs.** The 103 missing-primary
   training clips remain difficult. Further thermal work needs a new representation
   or pretraining rationale with gains across users, not another scan of existing
   seeds, crop parameters or fusion weights.

These are future priorities. This cleanup starts no new training and spends no
additional leaderboard submissions.

## Release and final candidates

Canonical strictV3 is unchanged. The recommended pair is `55712568` and `56036959`:
the verified baseline plus a parameter-free removal of weak Fusion. Both public
scores are 0.97512 and their predictions differ. The early 2,909-correct T+V OOF
used a Visual scale inconsistent with deployment; **that gain is not used to
justify selection**. The rationale is reproducibility, tied observed public
performance, a simpler model and limited prediction diversity, not guaranteed
private improvement. See [final submissions](../results/final_selection/README.md)
for downloaded CSV hashes, complete bundles and actual website status.
