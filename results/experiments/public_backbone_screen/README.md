# Public Depth/IR backbone screen

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

This study keeps the skeleton path conceptually unchanged and screens public
pretrained encoders as replacements for the old Visual/Fusion computation. It
does not load any project-trained checkpoint, inspect anonymous test data, or
use held labels for epoch selection.

The fixed screen uses users 1, 6, 17 and 22 as one held subject fold. Every
frozen encoder receives the same randomly initialized temporal probe for 25
epochs. Only the best Depth and IR family from that screen is fine-tuned for a
fixed 15 epochs.

| Public encoder | Sensor | Parameters | Frozen accuracy | Worst user |
| --- | --- | ---: | ---: | ---: |
| DeFM EfficientNet-B0 | Depth | 3.01 M | 0.25000 | 0.20197 |
| DeFM RegNetY-800MF | Depth | 6.25 M | 0.25279 | 0.21675 |
| DeFM ResNet-18 | Depth | 11.74 M | 0.25000 | 0.17734 |
| DeFM ViT-S/14 | Depth | 21.64 M | 0.25279 | 0.21182 |
| DFormerv2-S | Depth | 25.42 M | 0.28212 | 0.24631 |
| DFormerv2-B | Depth | 52.60 M | 0.26676 | 0.21675 |
| DFormerv2-L | Depth | 93.78 M | 0.21508 | 0.17734 |
| **Omnivore Swin-T** | **Depth** | **27.85 M** | **0.34777** | **0.25616** |
| **Omnivore Swin-T** | **IR** | **27.85 M** | **0.43296** | **0.33333** |
| M-SpecGene ViT-B | IR | 85.80 M | 0.24441 | 0.18301 |

Winner fine-tuning improves Depth Omnivore to **0.47207** and IR Omnivore to
**0.54469**. A fixed 25% Depth / 75% IR probability fusion reaches **0.55307**
with **0.43791** worst-user accuracy. Every naive feature concatenation
underperforms the frozen IR-only probe, so concatenation is rejected.

This is a screen, not an unbiased generalization estimate: the same held fold
was used to rank model families, and the 25%/75% diagnostic weight came from an
exploratory frozen-probe grid. Epoch counts and final checkpoints were fixed,
but a fresh untouched split or full OOF would be required before promotion.

All three DFormerv2 sizes were tested. Base and Large regress relative to
Small, so parameter count is not used as a proxy for transfer quality.

The research architecture is therefore: retained high-rate skeleton, one
fine-tuned Omnivore Depth branch, and one separately fine-tuned Omnivore IR
branch. Cross-attention remains a subsequent experiment, not a result inferred
from this screen. No candidate is promoted to `main`: these single-fold numbers
are selection-biased and not strong enough to replace the strictV3 release.

Raw features, logits and checkpoints live under the ignored
`runs/experiments/public_backbone_screen/` directory. The tracked
[`metrics.json`](metrics.json) records exact public-weight hashes and all
reported scores. Install optional dependencies with:

```zsh
python -m pip install -e '.[public-backbones]'
```

The full commands are exposed by:

```zsh
python -m yolo_r2plus1d.strict_v3.training.public_backbone_screen --help
python -m yolo_r2plus1d.strict_v3.training.finetune_public_omnivore --help
```

InfMAE could not be materialized from its official Baidu-only checkpoint,
Thermal-MAE is gated (HTTP 403), and no stable official DuGI-MAE checkpoint was
verified. ViT-Lens Depth was excluded because ViT-L is a poor use of the 100 MB
two-sensor package budget. The M-SpecGene release includes training state and
decoders (1.44 GB); only its verified 85.80 M-parameter encoder was evaluated.

The two Omnivore branches have an estimated 27.89 MB int4 parameter payload;
adding the retained skeleton/temporal/detector estimate gives about 38.54 MB
before serialization overhead. This passes only the preliminary arithmetic
screen. No official submission package has been built or size-certified.

Public sources: [DeFM](https://github.com/leggedrobotics/defm),
[DFormer](https://github.com/VCIP-RGBD/DFormer),
[Omnivore](https://github.com/facebookresearch/omnivore), and
[M-SpecGene](https://github.com/CalayZhou/M-SpecGene). Public weights are
downloaded into `.cache/` and are not redistributed by this repository.
Omnivore is CC BY-NC 4.0, so any released derivative checkpoint must preserve
its attribution and non-commercial restriction; this is also a release gate,
not merely a documentation note.
