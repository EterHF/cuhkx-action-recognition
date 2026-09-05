# Independent public sensor fusion

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

This study keeps a native-rate skeleton branch and independently fine-tuned
Omnivore Swin-T encoders for Depth and IR. All three target models were trained
from public initialization or from scratch; no historical project checkpoint
was loaded. The fixed held users are 1, 6, 17 and 22 (716 rows), epoch 30 and
seed 2026. The anonymous test set was not accessed.

| Candidate | Accuracy | Worst user | Decision |
| --- | ---: | ---: | --- |
| High-rate skeleton only | 0.455307 | 0.359477 | complementary, not standalone |
| Depth Omnivore | 0.472067 | 0.372549 | retain as independent diversity |
| IR Omnivore | 0.544693 | 0.418301 | strongest standalone sensor |
| Two clip tokens + cross-attention | 0.539106 | 0.411765 | reject |
| 8+8 temporal tokens + cross-attention | 0.515363 | 0.392157 | reject; over-fit |
| Equal Depth + IR + skeleton logits | **0.597765** | **0.503268** | research candidate only |

The temporal version preserves eight post-Swin time tokens per sensor and all
ordered skeleton frames up to 256. It nevertheless trained to 0.9651 accuracy
while generalising worse than clip-token fusion. The next architecture should
therefore keep independently supervised branch logits as the primary path;
cross-attention must be an optional residual, not a replacement classifier.

The equal-logit result is deliberately marked selection-biased because this
held fold had already been inspected when the ensemble diagnostic was run. It
is evidence of complementary errors, not promotable OOF and not a reason to
submit to Kaggle.

Mixed int8 packaging keeps backbone Conv/Linear tensors at per-output-channel
int8 and sensitive patch embeddings, relative-position biases, heads, norms
and biases at fp16. The serialized Depth and IR branches are 28,504,203 and
28,503,495 bytes, respectively (57,007,698 bytes total). Adding the retained
skeleton/temporal/detector estimate gives 67,657,705 bytes before final bundle
overhead. This is an arithmetic/package-component screen, not certification of
an official single-checkpoint submission. Under the same batch-16 inference,
mixed int8 changed Depth from 0.472067 to 0.467877 and left aggregate IR
unchanged at 0.544693.

Reproduce the fusion training and equal-logit diagnostic with:

```zsh
python -m yolo_r2plus1d.strict_v3.training.public_sensor_fusion --help
python -m yolo_r2plus1d.strict_v3.evaluation.public_sensor_ensemble --help
```
