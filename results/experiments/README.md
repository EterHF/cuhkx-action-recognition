# StrictV3 consensus experiments

[English](README.md) | [简体中文](README.zh-CN.md) | [Repository home](../../README.md) | [Technical report](../../docs/TECHNICAL_REPORT.md)

These candidates were recovered from the audited work in Codex thread
`01a04b41-e2ed-79c3-9fce-20eec40a3c73` and replayed with the current public
pipeline. They are deliberately separate from `checkpoints/strict_v3/model.pt`:
the `main` baseline remains the byte-reproducible 0.97512 release.

| Candidate | OOF | Delta | Fold gate | Single checkpoint | Test changes | Kaggle |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| [`sched30_consensus`](sched30_consensus/) | 0.960474 | +0.004282 | 5/5 non-degrading | 98,873,941 B | 3 | 0.97014 (ref 56016293) |
| [`temporal_pool_consensus`](temporal_pool_consensus/) | 0.959816 | +0.003623 | 5/5 non-degrading | 96,936,797 B | 12 | not submitted |

The first candidate averages probabilities from strictV3 and a fixed
three-seed sched30 temporal release. The second uses a majority vote over
motion-energy, top-2-frame and top-4-frame temporal pooling while sharing the
large model states.

The 0.960474 row is the frozen audited artifact. A fresh CPU retrain on
2026-09-03 produced 0.959486 after recalibration because a numerical grid tie
selected an adjacent weight. That rerun is documented as supporting evidence,
not as a replacement for this frozen candidate.

Audit both frozen candidates:

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.experiments.audit
```

Reproduce one candidate from raw test data and validate its frozen hash:

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.bundle \
  --model checkpoints/experiments/sched30_consensus.pt \
  --output /tmp/inference_bundle.pt
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --bundle /tmp/inference_bundle.pt \
  --output .cache/sched30_consensus.csv
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.experiments.audit \
  sched30_consensus --replayed-csv .cache/sched30_consensus.csv
```

No candidate is promoted or submitted based on anonymous-test inspection.
Promotion requires the frozen OOF gates, exact raw replay, the single-checkpoint 100 MB package
gate and an explicitly authorized Kaggle submission.

The rejected `highrate_cross_attention` study is retained as structural
evidence. Exact restored strictV3 branch OOF was Fusion 0.800395, Visual
0.896904 and Temporal 0.940053. Removing Fusion in a post-hoc diagnostic raised
the frozen blend from 2,903 to 2,909 rows. Native skeleton cross-attention did
not improve that Fusion-free baseline (2,904 rows), so no test inference was
performed. See [`highrate_cross_attention/metrics.json`](highrate_cross_attention/metrics.json).

The [`public_backbone_screen`](public_backbone_screen/) study evaluates public
Depth/IR pretrained encoders without any project checkpoint. Omnivore Swin-T
wins both sensor tracks, but the fixed-fold result is not strong enough for
promotion to `main`.

The follow-up [`public_sensor_fusion`](public_sensor_fusion/) study retains
native-rate skeleton plus independent Depth and IR Omnivore branches. Learned
clip-token and temporal-token cross-attention both failed, while a diagnostic
fixed equal-logit ensemble reached 0.597765 on the reused held fold. It remains
selection-biased evidence pending a fresh full OOF.

The authorized [`public_sensor_full_fit`](public_sensor_full_fit/) deployment
retrained all three branches on 3,036 rows and reached 0.950593 in-sample with
the actual 57.93 MB mixed-int8 bundle. Its Kaggle public score was only 0.58706
(ref 56035438), so the route is closed without leaderboard-driven reweighting.

The [`temporal_public_sensor`](temporal_public_sensor/) follow-up anchors on
strictV3 Temporal and adds fixed 5% Depth and IR probability residuals. Complete
five-fold OOF fell from 0.940053 to 0.939723, so the sensor gate remains zero and
no test candidate was built.

The [`omnivore_layer_fusion`](omnivore_layer_fusion/) screen fuses all four
Depth/IR Omnivore stages with bidirectional temporal cross-attention and one
classifier. Fold-A held accuracy was 0.512570, below independent IR and the old
Fusion branch, so it was stopped before B–E or test inference.

The [`omnivore_native_rgbd`](omnivore_native_rgbd/) screen activates one
Omnivore trunk's native four-channel token fusion with repeated IR plus
inverse-depth. Fold-A held accuracy was 0.379888, exposing a strong mismatch
with its natural-RGB plus metric-depth pretraining contract; the route is closed.

The [`temporal_visual_equal`](temporal_visual_equal/) ablation removes Fusion
and combines frozen-temperature Temporal/Visual logits at exact 50/50 weights.
OOF fell to 0.948946 and Kaggle scored 0.95522 (ref 56036875), so strictV3 is
unchanged.

The [`temporal_visual_inherited`](temporal_visual_inherited/) optimization sets
only Fusion's frozen base weight to zero while inheriting the strictV3
Temporal/Visual temperatures, weights and quality gate. OOF improved 6 rows with
5/5 folds non-degrading, and Kaggle tied 0.97512 (ref 56036959).

The [`visual_priorities`](visual_priorities/) study sequentially adds a
layer2/3/4 temporal residual head, high-resolution layer4 time stride, and a
zero-initialized Depth/IR gate to the Visual R(2+1)D-34. Fixed Fold-A accuracy
was 0.625698/0.608939/0.618715/0.617318 for control and the three cumulative
arms. None exceeded control, so no test inference or submission followed.

The [`quantization_budget`](quantization_budget/) study physically removes
Fusion and inactive thermal weights, producing a 50.73 MB T+V single
checkpoint with identical predictions. Raising the main Visual from 5-bit to
6/8-bit gives no final OOF gain. A traceable NTU120 proxy confirms that INT4
can destroy external-pretraining information and that fresh 5-bit quantization
can recover it.

The [`conditional_corrector`](conditional_corrector/) implementation adds a
small zero-initialized Visual residual conditioned on Temporal logits. Its
strict external-only nested OOF corrected a net 264/3,036 rows with positive
changes in all folds and users. Because the proxy branches are much weaker and
differently ordered than strictV3, this validates the mechanism but does not
authorize deployment or submission.
