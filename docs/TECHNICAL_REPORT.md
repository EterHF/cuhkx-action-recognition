# CUHK-X Small Model Track — Technical Report

[English](TECHNICAL_REPORT.md) | [简体中文](TECHNICAL_REPORT.zh-CN.md) | [Documentation index](README.md) | [Repository home](../README.md)

**Best public Kaggle score so far: `0.97512` (rank 3, ref `55712568`)**<br>
**Gap to rank 2: `0.00497` (one public sample's net swing).**<br>
**Frozen: 2026-09-04**

> This report folds the historical improvement line and the next-step
> strategy into a single document. It is the only project-level technical
> narrative; the per-candidate submission log, the strict-v3 priorities
> memo, the cleanup manifest, and the next-step directions from the
> `01a04b41…` codex session are summarised here and no longer shipped as
> separate files.

---

## 1. The problem and the data

CUHK-X Small Model Track ([challenge page](https://openaiotlab.github.io/CUHK-X-Challenge/),
UbiComp / ISWC 2026) is a 40-class, cross-user action-recognition task with
six modalities per clip:

- 16 ordered **Depth Color + IR** frames at 128×128. YOLO probes 8 IR frames
  to define one clip-level person window; Thermal has zero release weight.
- Timestamp-sorted **IMU** sequences (temperature / battery / firmware
  metadata stripped).
- **Skeleton** (position, root motion, relative pose, velocity, acceleration).
- **Radar** frame statistics (empty-header files treated as missing).

The training split has **3,036 clips, 18 users, 40 classes**; the public
test split has 405 paths. The user-level partition is the only valid
generalisation unit — random clip splitting is forbidden because of 1,043
near-duplicate trial groups (964 of them triples). Class imbalance is about
30× (12–365 clips per class).

Three disjoint user folds score 53.96%, 48.53%, and 50.31% (mean 50.93%,
std 2.26%) on the original EfficientNet-B0 + 2.30M-parameter temporal
fusion head baseline. The final three-seed ensemble is 49.17 MB, well
under the 100 MB Small Track budget.

---

## 2. The two "strict" definitions

Throughout the report and in every experiment, the two meanings of
"strict" must be kept apart:

| Name | Definition | What it can support |
| --- | --- | --- |
| `release-strict` | The strict-v3 release plan. Uses public checkpoints; *some* of those checkpoints saw CUHK-X target labels during their upstream training, so outer held-users may have leaked into their pre-training. | Current submission and engineering replay baseline. **Cannot** be quoted as fully-leakage-free OOF. |
| `fully-external-pretrained strict` | Initialisation comes only from Kinetics / NTU / PKU-MMD. For every CUHK-X outer fold a fresh 40-way head is trained only on the outer-train users; preprocessing, epoch selection and hyperparameters are also fixed on outer-train. | Cross-user generalisation comparison and method-selection decisions. |

The `release-strict` 95.619% OOF is the *current submission evidence*;
it is not a fully-leakage-free baseline. The "research" branch lives
under `fully-external-pretrained strict`.

---

## 3. The current best: `legal_strict_v3`

The `legal_strict_v3` submission reached **0.97512** (rank 3, ref `55712568`).
The current package recreates that scored CSV byte-for-byte from raw test data.
The key deployment fix is an explicit `visual_package_output_scale=0.5`: one
retained 5-bit visual member represents its half-weight contribution without
storing a second near-duplicate network.

| Property | Value |
| --- | --- |
| Official-format inference bundle | `checkpoints/strict_v3/submission_bundle.pt` (69,805,793 bytes) |
| Auditable model source | `checkpoints/strict_v3/model.pt` (64,206,000 bytes) |
| Canonical submission | `results/strict_v3/submission.csv` (405 rows, 40 classes) |
| Raw-replay submission | `results/strict_v3/raw_replay/submission.csv` (405 rows, 40 classes) |
| All inference weights in one checkpoint | 69,805,793 bytes (30,194,207-byte margin) |
| Public score | **0.97512**, ref `55712568` |
| Reproduced raw-data score | **0.97512**, byte-identical to ref `55712568` |
| OOF (release-strict) | aggregate `0.956192`, mean fold `0.954823`, worst fold `0.926931`, macro recall `0.953143` |
| Protocol | `five_fold_subject_wise_nested_temperature_quality_gate` (full contract in `release_manifest.json`) |
| Raw-replay evidence | deterministic SHA-256 `e2509491…`; byte-identical to the canonical CSV |

The organizer's [official ensemble clarification](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/729056)
defines size on one checkpoint containing every weight required at inference,
including every ensemble member. It must be **under** 100 MB on disk; FP16 and
INT8-or-lower quantisation are allowed. The release therefore embeds the exact
YOLO file bytes in the same weights-only-loadable checkpoint as strictV3 rather
than relying on the earlier sum-of-two-files interpretation.

### 3.1 What the strict-v3 release actually contains

The release is a three-branch multibranch fusion whose inference contract
is recorded in `release_manifest.json`:

* **Visual head**: train-only affine preprocessing, with no per-clip z-score.
* **TCN / temporal head**: per-fold-held OOF logits replayed through a
  fixed recipe.
* **Fusion head**: the legacy multibranch Fusion4 logits, also replayed.

The first cleanup package incorrectly replaced the source 5/6-bit visual
ensemble with two 4-bit members. Its deterministic replay changed
`SM_test_0214` and `SM_test_0378` and scored 0.97014 (ref `55978481`). The
current package restores the previously audited compact representation and
eliminates both differences while remaining well below the size limit.

The best public history, in one table:

| ref | candidate | public score | status |
| --- | --- | ---: | --- |
| 55613900 | original baseline | 0.46268 | first baseline |
| 55620179 | YOLO + R(2+1)D-18 | 0.60696 | first visual model |
| 55673502 | R(2+1)D-34 int5/int6 + YOLO11n | 0.71641 | base visual candidate |
| 55709862 | Fusion4 / TCN (legal multibranch) | 0.89552 | exploratory; rejected — package replay + raw replay agree but public split generalises poorly |
| 55712568 | **legal_strict_v3** | **0.97512** | **current best** |
| 55714393 | strict adapter v1 / v2 | 0.96517 | below strict-v3; rejected |
| 55834961 | P0.3 visual mix (base + seed-2027 adapter) | 0.96517 | below strict-v3; rejected |
| 55846236 | strict seed-pair 2027/2028 P0 | 0.94527 | rejected — public split generalisation failed |
| 55857342 | strict top-2-frame-mean temporal pooling | 0.94029 | rejected |
| 55858076 | strict top-2 v2 | 0.94029 | rejected |
| 56006027 | strictV3 0.90 + PKU bridge INT4 0.10 | **0.97512** | tied canonical; confirmatory external-data diagnostic, not promoted |
| 56014518 | strictV3 0.90 + NTU120 INT4 0.10 | **0.97512** | tied canonical; larger-NTU confirmation, not promoted |
| 56016293 | strictV3 0.50 + sched30 three-seed consensus 0.50 | 0.97014 | offline OOF improved, but public generalisation failed; rejected |
| 56035438 | full-fit equal Depth Omnivore + IR Omnivore + skeleton, mixed INT8 | 0.58706 | severe cross-user generalisation failure; rejected |
| 56036875 | strictV3 Temporal + Visual exact 50/50 | 0.95522 | below strictV3; rejected |
| 56036959 | strictV3 Temporal + Visual inherited gate | **0.97512** | tied canonical; simpler private-LB candidate |

Multiple offline "higher OOF" candidates (Fusion4 raw at 0.970*, etc.)
were **refused by the public leaderboard**; they are no longer candidates.

### 3.2 Current-best data and training flow

The scored current best remains `legal_strict_v3`; the strongest train-only OOF
candidate is its fixed 50/50 probability consensus with `sched30`.
The shared data path is:

1. Discover clips in a deterministic order and retain only class and subject
   metadata required for subject-wise CV.
2. Probe 8 uniformly spaced IR frames with YOLO11n. Expand the union person box
   by 1.4 with a 0.35 minimum side; fall back IR → Depth Color → full frame.
3. Uniformly sample 16 aligned Depth Color and IR frames, resize the shared crop
   to 128×128, and store a uint8 `[T,4,H,W]` cache. Fit the affine mean/std only
   on the relevant training users; never fit per-clip or test statistics.
4. Build the aligned pelvis-centred H36M-17 skeleton cache and the 16-frame
   crop-scaled DSTFormer inputs. Missing skeletons affect encoder input only,
   never a post-hoc blend mask.
5. Infer visual R(2+1)D, visual+skeleton Fusion, and DSTFormer→TCN branches.
   Apply outer-train-only temperature calibration and the fixed quality gate;
   full-data release weights are Fusion 0.11, Visual 0.22 and Temporal 0.67.

Training uses five disjoint held-subject folds. Visual head adaptation uses
cross-entropy with 0.02 label smoothing and train-only horizontal flips. Fusion
uses the same visual cache plus skeleton noise 0.005, separate visual/new-layer
learning rates and a OneCycle schedule. The temporal residual TCN uses AdamW
(3e-4, weight decay 0.01), 0.01 label smoothing, reversal probability 0.25 and
Gaussian logit noise 0.01. `sched30` trains fixed epoch 5 under a 30-epoch cosine
schedule for seeds 2026–2028, averages their probabilities, then takes a fixed
50/50 probability mean with strictV3. It improves release OOF from 2,903 to
2,916/3,036 with 5/5 non-degrading folds, occupies 98,873,941 bytes as one
checkpoint, and reproduced CSV SHA-256 `f33e0569…` from raw data again on
2026-09-04. It is stronger offline, but its frozen Kaggle submission scored
0.97014 (ref 56016293), below the 0.97512 public baseline.

### 3.3 How to reproduce `legal_strict_v3`

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --output results/strict_v3/reproduced_submission.csv
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify \
  --replayed-csv results/strict_v3/reproduced_submission.csv
```

The verifier requires the raw replay to be byte-identical to the canonical
CSV. The recipe is recorded in `release_manifest.json`; any change to the
inference contract requires a new package and manifest.

### 3.4 Retraining audit on the cleaned repository

On 2026-09-03 the cleaned training entry points were exercised with new model
updates rather than checkpoint-only replay. The strict temporal branch matched
all five historical fold accuracies (aggregate 0.940053). The fusion refinement
matched all five historical best fold accuracies when its actual batch-8
contract was restored. The visual head matched folds A, B and E exactly by
accuracy; folds C and D differed by −3 and +2 correct rows respectively.
Non-deterministic cuDNN kernels and loader-worker random streams explain why
the newly trained logits are not byte-identical even where accuracy matches.

The three-seed `sched30` suite was also retrained. Its temporal mean reached
0.937747 OOF; recalibration with the frozen visual/fusion branches reached
0.959486 (+0.003294 over strictV3, 5/5 folds non-degrading). This is distinct
from the frozen 0.960474 consensus package: a CPU/GPU numerical tie selected
an adjacent calibration weight. Neither result changes the canonical release.
The retrain remains partial until a new full fusion package, two byte-equal raw
replays and an authorized Kaggle confirmation all pass. Detailed results and
acceptance criteria are in [`RETRAINING.md`](RETRAINING.md).

---

## 4. The historical improvement line (every direction we tried)

This section is the audit chain. Every direction is listed with its
result, the reason it stopped, and the canonical evidence file. Most of
these individual recipes ended up *exhausted* (further tuning of the same
recipe would only inflate the validation-set search space). That label does
not close a data source when a genuinely different, preregistered transfer
mechanism remains untested.

### 4.1 Original EfficientNet-B0 baseline (historical)

The EfficientNet-B0 multimodal baseline plus the auxiliary-loss
selection fold (`pretrained_aux015`, mean OOF ≈ 50.93%) and the
three-seed ensemble (`pretrained_final_seed{2026,3407,8819}`,
`pretrained_ensemble/`) formed the original **release-strict** anchor. Those
weights and scripts are not shipped on the cleaned main branch; the measured
outcome is retained here for provenance.

### 4.2 Visual / temporal / fusion branch development (kept as evidence)

The YOLO-cropped R(2+1)D-18 → R(2+1)D-34 int5/int6 + YOLO11n ladder
brought the visual-only candidate from 0.46268 → 0.71641. Adding the
ST-GCN/DSTFormer-style skeleton fusion and the temporal TCN residual
heads produced the legal "5-bit multibranch" package. The strict-v3
release is the result of multiplying these branches by a per-fold nested
selection rule (no leaderboard feedback, no sample/user ID leakage).

### 4.3 External-data scaling (measured; target confirmation reopened)

NTU Depth / Skeleton pretraining and PKU-MMD bridge were explored
heavily. The final local inventory audit found 248 GB of NTU material (all 32
masked-depth setup archives, both skeleton archives, but only 9 IR setup
archives) and 156 GB of PKU-MMD Phase 2 material (6,952 trimmed depth records,
13 inferred subjects and 41 classes). Thus the available data support honest
depth / skeleton studies, but not a claim of complete paired NTU Depth+IR
coverage. More source data can improve representation coverage, but volume
alone is not a mathematical guarantee: domain, modality and label mismatch,
plus catastrophic forgetting, determine whether that information survives
target adaptation. In this repository the evidence already contains strong
positive transfer, so the data source itself must not be described as
exhausted. The measured transfer summary is:

| Experiment | Result | Verdict |
| --- | ---: | --- |
| NTU masked-depth pretraining + R(2+1)D-18, 3 seeds × 5 folds | mean micro 63.669% versus Kinetics 60.299%; non-degraded folds per seed 5/4/5 | Passed the frozen transfer gate; real positive signal at small backbone |
| NTU60 / NTU120 direct R(2+1)D-34 supervision | 58.56% / 58.89% | Large source-domain supervised training **forgets** the Kinetics representation |
| Kinetics-anchor + 25% NTU60 R(2+1)D-34 interpolation, 3 seeds × 5 folds | cell-mean 65.760%; matched mean +5.033 pp; 15/15 cells non-degraded | Strong, reproducible target transfer, though absolute standalone accuracy was not release-competitive |
| Same NTU60 interpolation + true epoch-1 head-only progressive unfreeze | paired single-seed mean 66.0848% versus 66.0518% (+0.033 pp); worst-user mean −0.152 pp | Failed the preregistered stability / worst-user gate; stopped before fusion or deployment |
| NTU Depth+IR on the 9 complete local setups | fold C/E 63.62% / 61.59% versus depth-only 68.04% / 61.38% | Mixed partial-data result (C −4.42 pp, E +0.21 pp); it neither proves nor disproves benefit from complete paired IR, which needs a separate frozen experiment |
| NTU Skeleton student | source val 64.68%; target A–E micro 46.81%, macro 39.51%, worst-user 25.00% | Source fit did not survive target cross-user validation |
| Earlier unconstrained PKU-MMD source-only pretraining | source val 75.30%, source train ~99% at epoch 15 | This 15-epoch recipe over-fit; source validation does not proxy target transfer |
| Earlier Kinetics → PKU → CUHK-X | 61.92% (control 63.27%) | Direct continued-training caused **negative transfer**; this rejects the recipe, not PKU-MMD |
| Kinetics+NTU60 → constrained PKU layer4 bridge, source-subject CV | 63.828% versus 57.892% control (+5.936 pp); 9/9 seed-fold cells positive | Passed every frozen source gate and justified one separate target-domain confirmation |
| Same bridge, target-domain 3 seeds × 5 folds | mean micro 67.7866% versus 66.9521% (+0.8344 pp); every seed positive; diagnostic logit mean 68.5441% | Real target transfer, but rejected for 3/5 fold stability in seed 2027 and a 2.5339 pp worst-user cell drop |
| Fixed seed-2028 bridge, uniform INT4, strictV3 90/10 deployment diagnostic | external OOF 63.5705%; blended OOF 95.5204%; Kaggle 0.97512 (ref 56006027) | Passed the diagnostic gate and tied canonical; no accuracy gain, no promotion or leaderboard retuning |
| PKU visual replacing strict-v3 visual | 95.191% (control 95.619%); nested mean weight = 0 | Cannot serve as a 4th logit branch |

The progressive-unfreeze result tested a target-training scope change on top
of an already useful NTU initialization; it did not test whether NTU data were
useful. Likewise, the earlier PKU failures used different, less constrained
recipes. The seed-matched PKU bridge confirmation in §5.7 therefore changes
only the external initialization and retains the exact target recipe of its
immutable matched control.

### 4.4 Head-only / L2-SP / SAM / EMA / temporal-diff (exhausted)

Every release-strict inner CV exhaustively tested these mechanisms on
the full A–E matrix:

| Direction | Outcome | Reference |
| --- | --- | --- |
| Head-only fine-tune (full A–E) | micro 0.146684 — severe underfit | `audit_head_only_scope_full_ae_v1.py` |
| L2-SP λ = 0.01 (full A–E) | micro 0.664141, delta −0.005 — no release | `audit_l2sp_temporal_difference_full_ae_v1.py` |
| Canonical SAM ρ = 0.05 (C/E) | micro 0.585780, delta −0.076 — no expansion | `audit_sam_temporal010_screen_v1.py` |
| Success-step EMA 0.995 (C/E) | micro 0.662061 — no expansion | `audit_success_step_ema_temporal010_v1.py` |
| Temporal-difference w = 0.10 (full A–E) | micro 0.669521 — failed all-seed / worst-user / generalisation-gap gates | `audit_temporal_difference_weight010_v1.py` |

None of these were promoted.

### 4.5 Multirate / R2D18 / residual / TSM-historical / category-pair / TTA / calibration / margin-arbitration / quantization-sweeps (exhausted)

These were all rejected on the strict inner CV:

* **Multirate (continuous skeleton+IMU) branch fusion**: every small
  weight on fold C causes regression; fold E improves by at most ~1%
  but the two-fold non-degradation rule is violated. Permanently stopped.
* **Strict R2D18 single-modality ensemble**: 3-seed stability fails;
  every fold shows 1–6 OOF samples worse than the baseline.
* **Residual / boosted residual multi-window fusion**: fold C still
  regresses by 3–4 rows; the representation is the bottleneck.
* **TSM-MobileNet historical C/E (fixed epoch 3)**: 0.280 / 0.232 —
  far below any usable complementary interval. No more training.
* **Nested top-2 category-pair correction**: the inner selection
  collapsed it to zero changes; the held/validation tag leakage
  prevented any meaningful signal.
* **Fixed margin arbitration / calibration / TTA / multi-seed
  quantization sweep**: all converge to the same failed CSV; the
  maximum OOF is 0.95883, below the strict-v3 candidate. Path is
  exhausted.

These were the *穷尽* (exhausted) directions.

### 4.6 Compliance / rules boundaries (kept as evidence)

* **LLM use rule**: only prediction-time LLM use is forbidden; AI coding
  assistants are allowed. This was confirmed via a Kaggle discussion reply
  and retained in the historical audit before cleanup.
* **External-data competition rule**: the organiser's
  [official clarification](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/724404)
  permits public external datasets and pretrained models when they are freely
  or reasonably obtainable and disclosed in the final writeup. A request form
  open to anyone is acceptable, and NTU RGB+D is explicitly permitted. This
  is recorded in host comments `3495517` (2026-07-12), `3496001`
  (2026-07-13), and `3504934` (2026-07-29), and supersedes the later local note
  that incorrectly treated permission as unresolved.
* **NTU RGB+D terms**: the [official provider page](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
  limits the dataset to academic research and restricts redistribution and
  commercial use. Local archives and NTU-derived research checkpoints remain
  outside the open-source package.
* **PKU-MMD licence**: the [official project page](https://struct002.github.io/PKUMMD/)
  directly publishes the research data but does not state a separate explicit
  dataset licence. That residual issue is disclosed and raw data are not
  redistributed; it is not treated as a competition-training ban. Any derived
  weight proposed for public release still requires a separate clean-release
  review.
* **Pretraining and distillation**: the organiser's
  [official reply](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/711665)
  allows a small standard pretrained model such as ResNet-18 and allows
  knowledge distillation. MViTv2-S's ≈131.9 MB checkpoint cannot be the final
  Small-Track artifact under the ≤100 MB rule, but it may be investigated as a
  training-only teacher when the submitted student and complete package obey
  the final-model rules.
* **TorchVision licence reminder**: pretrained weights may inherit
  training-data terms; documented in the External-Compliance review.
* **Finalist source licence**: the
  [official challenge page](https://openaiotlab.github.io/CUHK-X-Challenge/)
  says Top-6 finalist solutions must be released under Apache-2.0 within 30
  days of the finals. This repository is currently MIT-licensed. A maintainer
  who owns the relevant copyrights must confirm and perform any relicensing
  before a finalist release; an automated cleanup must not silently change
  third-party or contributor rights.

---

## 5. Exploration ledger and frozen follow-ups

These directions are an archival research queue. They are recorded here so
the rationale survives cleanup; their preregistration, experiment and protocol
code is deliberately not shipped on the strictV3-only main branch.

### 5.1 Strict nested selection of candidate fusion (priority 1)

* **Motivation**: every *hand-picked* coefficient / leaderboard-driven
  single-row tweak has already been refused. The only admissible
  selection rule is "each held-fold coefficient is determined by the OOF
  on the other four folds only — no user / sample IDs, no test
  attributes".
* **Already implemented (CPU / synthetic)**: Fusion7 nested selection
  (OOF +14 rows, 5/5 non-degrading, natural 40-class coverage);
  50/50 probability-mean vs. strict-v3 (5/5 non-degrading, +9 net
  rows, 1 row short of acceptance); nested-coefficient grid
  (end-points + 0.25 / 0.5 / 0.75); shared-state multi-pooling candidate
  (energy + top-2, reusing the same Fusion4 / Visual4 / DSTFormer,
  OOF +12 net rows, 5/5 non-degrading, natural 40-class coverage,
  historically estimated at 86.43 MB / 92.05 MB with YOLO by component
  accounting; this was not an official single-checkpoint measurement).
* **2026-09-04 close-out**: the final `sched30` package was materialized as one
  98,873,941-byte checkpoint and replayed from raw data through its embedded
  YOLO bytes. The resulting CSV was byte-identical (`f33e0569…`) to the frozen
  candidate. Its subsequent frozen Kaggle submission scored 0.97014 (ref
  56016293), so it was not promoted. The broader historical Fusion7 precision
  mismatch remains an archived direction, not a release claim.
* **Next-step design points**:
  1. The selector reads only OOF logits of the other four folds and
     marginal / agreement signals. Forbidden: user / sample-level
     attributes.
  2. The selection rule is pre-declared. No "look at fold C, then
     adjust threshold" patterns.
  3. Only after two-round raw-replay agreement and the complete
     nested-gate pass may the candidate be added to the submission
     queue.

### 5.2 Equal-weight probability consensus / multi-pooling (priority 2)

* **Motivation**: with every quantisation / calibration / TTA path
  exhausted, *same-weight different-pooling* still changes a small
  number of errors and adds no new weights (keeps the 100 MB budget).
* **Already implemented**:
  * `sched30` three-seed temporal consensus (FP32): OOF +0.004282, 5/5
    non-degrading, every seed 5/5, 98.87 MB as one checkpoint, CSV
    `f33e0569…`; Kaggle public 0.97014 (ref 56016293), rejected for promotion.
  * Temporal-pooling majority vote (top-2 + energy): OOF +0.003623, 5/5
    non-degrading, 96.94 MB as one checkpoint, CSV `6a320486…`.
  * A fixed three-way majority over strictV3, `sched30` and temporal
    pooling reached only 2,909/3,036, below `sched30` at 2,916/3,036. Nested
    confidence/margin/entropy routing and class-prior correction also failed
    to exceed the fixed `sched30` consensus, so no test inference followed.
* **Next-step design points**:
  1. Treat the two pooling variants as a structural minimum, and
     perform strict OOF nested selection — *not* a fixed 50/50 blend.
     No manual coefficient picking.
  2. New pooling variants are accepted only if they share the
     DSTFormer frame logits with the anchor / `sched30` and add no
     large weights.
  3. Every candidate must produce two byte-equal CSVs and a deployment
     logits that matches the training OOF exactly in FP16/FP32.

### 5.3 From-scratch small backbone — "competition-only" (priority 3, diversity fallback)

* **Motivation**: a from-scratch small model provides a genuinely different
  inductive bias and a clean provenance fallback. It is not required because
  external training data are forbidden; they are allowed. The deployed model
  must still satisfy the Small Track ≤100 MB rule, and MViTv2-S itself is too
  large to be the final artifact.
* **Candidates** (preregistered and audited at the CPU / contract
  level, 87/87 contract + 20/20 signal tests passing):
  * **Primary**: `TSM-MobileNetV3-Small`, 975,576 parameters,
    FP32 `state_dict` ≈ 4.03 MB; explicit `weights=None`, no external
    download. `no_shift_meanmax` (C0) vs. `tsm_meanmax(div=8)` (C1),
    both arms share the same per-tensor init at every seed.
  * **Backup**: `S3D`, 7,954,184 parameters, FP32 ≈ 32.07 MB, also
    `weights=None`.
* **Frozen recipe (no tuning, no held-based epoch selection)**:
  * TSM: 100 epochs, SGD momentum 0.9, LR 0.01 (target effective
    batch 64), weight decay 1e-4, LR decay 40/80. The model head, BN,
    mean+max, edge-keeping and cache augmentation are kept as the
    project's adapter state. **No claiming a paper recipe.**
  * S3D: single fixed bundle, no grid search.
* **Next-step design points**:
  1. **Single-process paired training**: one job, one augmented batch,
     trains C0/C1; reload after delete→rebuild before constructing the
     held loader; the held loader is iterated exactly once.
  2. The candidate space is frozen as `C0/C1/C2`; S3D is *not* added
     after seeing TSM.
  3. Split: 2 inner seeds × 4 inner folds (≈ 80 process jobs);
     afterwards outer ≤ 30 one-shot jobs; every job audited P0/P1 = 0.
  4. Outer validation uses *per-fold local* selection. Only when 5/5
     outer folds select consistently is the candidate allowed to be
     promoted to a new outer preregistration.
  5. The shared builder must run inside the *build→audit→launch*
     exclusive single-writer window. The TSM runner must not leak
     C1 metrics into outer.
  6. Input wiring is unified to **input snapshot + SHA manifest**:
     `authorization` ≥ 21 rows; logical paths only for contract
     comparison, I/O only through private snapshot.

### 5.4 External representation transfer (active under frozen gates)

The compliance correction reopens bounded external-data work without reopening
post-hoc hyperparameter searches. The queue is:

1. **PKU-MMD bridge target confirmation (completed and rejected in §5.7)** —
   one seed-matched initialization comparison against an immutable control;
   the average transfer was positive, but the conjunctive stability gate
   failed. No source-weight, LR or epoch scan is allowed.
2. **VideoMAE-S ≈ MViTv2-S** — same investigation tier; the "≈" means
   "same investigation tier", not "equivalent on this dataset".
   Freeze the external checkpoint / architecture / preprocessing / seed
   map before opening any CUHK-X fold.
3. **External IR closed for this report** — NTU has only 9/32 complete local IR
   setups, while the local PKU-MMD tree contains 902,397 depth PNGs but no IR
   archive or extracted IR tree. The attempted NTU completion would require a
   further 320.8 GB and was stopped. No incomplete paired-IR result is used;
   external transfer claims are explicitly depth-only.
4. **Deterministic depth / lag-1 temporal-difference channels** —
   fixed, auditable; pre-declared auxiliary weight. No held-driven
   channel-recipe search.
5. **Predeclared tail weight averaging (executed; rejected)** — the fixed
   epoch 3–5 interval, LR schedule, parameter scope and BN treatment were
   declared before A–E. The averaged checkpoint was the only candidate;
   results are frozen in §5.5 and no post-hoc SWA schedule scan is allowed.
6. **SlowFast / X3D** — same subject folds, seed budget, source-only
   data boundary; measure compute / accuracy locally first.
7. **MixStyle / ASAM** — last in queue, because small-fold spurious
   gains tend to damage cross-seed stability.

External-only is *strict*: external representation / public
architecture / deterministic training transform. **Never** the test /
anonymous labels / IDs / samples / submission scores / prediction
history / post-hoc leaderboard feedback.

### 5.5 Temporal generalisation checks, 2026-09-03 (completed; rejected)

Three one-variable candidates were locked before their A–E runs. All used the
same train frame-logit SHA `76e6c11…7102e`, metadata SHA `bf2e93e5…9f099`,
seeds 2026/2027/2028, five fixed sched30 epochs and deferred held metrics.
Each suite ran in OOF-only mode: 15/15 jobs completed, no full-data model or
test file was created, and every receipt recorded `test_data_loaded=false`.

| Candidate | Temporal result vs fresh sched30 | Generalisation gate | Release gate / decision |
| --- | --- | --- | --- |
| Epoch-varying deterministic reversal/noise | 2,847/3,036 = 0.937747; seed 2026 gained one row but the seed mean changed zero predictions | macro, subject-macro, worst-user, worst-fold and 5/5 folds tied; failed the preregistered +1-row minimum | Stopped at stage 1; rejected |
| Uniform FP32 TCN parameter average, post-update epochs 3–5 | 2,848/3,036 = 0.938076 (+1 row); macro +0.000327; subject-macro +0.000208; worst metrics tied | 4/5 folds and 3/3 matching seeds non-degrading; passed stage 1 | nested branch 2,920/3,036 = 0.961792 (+1 vs historical branch), but fixed 50/50 strictV3 consensus stayed 2,916/3,036 = 0.960474 with zero changed predictions; required 2,917, so rejected at stage 2 |
| Same-class cross-user temporal-residual mix, fixed 0.25 | 2,847/3,036 = 0.937747; all three seed scores and 5/5 fold scores tied; zero top-1 changes | macro, subject-macro, worst-user and worst-fold tied, but the +1-row minimum failed | Stopped at stage 1; rejected; no strength/probability scan |

The train-only nested evaluator was first checked against the historical
sched30 inputs: both the saved nested logits and the final probability array
were exactly equal (`max_abs=0`). Evidence hashes are: epoch-resampled receipt
`dd135d65…df24`, OOF `0ddb3bdd…67c0`; tail-average receipt
`40510e90…26ac`, OOF `99a0df0b…5960`; final nested-gate metrics
`d8edb37a…044b2`. For the data/training candidate, every outer-train class had
at least two users. A deterministic same-class partner from a different
outer-train user contributed 25% of the zero-mean 16-frame residual while the
anchor clip mean stayed unchanged. All 15 checkpoints differed from control
and the OOF logits changed (`max_abs=0.115523`), proving that the treatment ran,
but none of 3,036 decisions moved. Its receipt is `8f1e8612…787b`, OOF is
`1cb107ed…6a12`, and rejection decision is `7e20e00e…a01a`.

The stop rules prohibited full-data training, anonymous-test inference,
packaging and coefficient changes after seeing A–E. In accordance with
repository cleanup policy, the rejected mechanisms remain documented here
rather than as permanent training flags or model files.

### 5.6 NTU progressive-unfreeze revisit, 2026-09-03 (completed; rejected)

The historical `head_warmup_epochs=2` grid only ramped the fresh head's
learning rate; it did **not** freeze layer4. One genuinely different variable
was therefore locked before training: initialise R(2+1)D-34 from the frozen
Kinetics + 25% NTU60 masked-depth encoder, train only the fresh 40-way head in
epoch 1, then train layer4 + head in epochs 2–15. Cache, affine contracts,
optimizer groups, cosine schedule, augmentation, BN policy, folds, seeds and
fixed-final checkpoint selection were identical to the matched control.

All 15 jobs (3 seeds × 5 subject folds) completed on GPU. The runner asserted
that layer4 was bitwise unchanged while the head changed in epoch 1, and that
layer4 changed after unfreezing. Held labels were evaluated once only after
each epoch-15 checkpoint had been written and reloaded.

| Frozen endpoint | Control | Progressive | Delta / gate |
| --- | ---: | ---: | --- |
| Mean paired single-seed micro | 0.660518 | 0.660848 | +0.000329; required ≥ +0.002 |
| Mean subject-macro | 0.653176 | 0.654989 | +0.001813; passed direction only |
| Per-seed micro delta (2026 / 2027 / 2028) | — | — | −0.010870 / +0.009223 / +0.002635 |
| Joint micro+subject non-degraded folds per seed | — | — | 2 / 4 / 3; required ≥ 4 for every seed |
| Mean worst-user delta / largest cell drop | — | — | −0.001517 / −0.035461; both failed |
| Three-seed logit mean (diagnostic, not the gate) | 0.669960 (2,034 rows) | 0.672596 (2,042 rows) | +8 rows, but worst-user 0.44375 → 0.425 |

Only the subject-macro direction passed; five of six preregistered criteria
failed. The candidate was therefore rejected without a warmup-length or LR
scan, strictV3 fusion, full-data training, anonymous-test access, checkpoint
packaging or Kaggle submission. The frozen evidence identifiers are:
preregistration `1b21f4ef…ddadc`, runner `c923a4d9…f556`, summary
`ee2fc114…49380`, decision `57776a38…f949bd`, and seed-mean OOF logits
`a339f104…3cb10`. This rejects only progressive unfreezing. PKU-MMD was not
part of that treatment; the prior statement that its competition permission
was unresolved was incorrect and is superseded by the official organiser
clarification in §4.6.

### 5.7 PKU-MMD bridge target confirmation, 2026-09-04 (positive transfer; promotion rejected)

The organiser clarification and the already approved 2026-08-30 access review
allowed the previously materialised PKU-MMD bridge to receive one fresh target
confirmation. No incomplete or quarantined prior target outcome was opened.
Before training, the experiment froze the 15-job matrix, all input hashes and a
conjunctive gate. The sole treatment variable was encoder initialization:

* control: Kinetics anchor + 25% NTU60 masked-depth interpolation;
* treatment: the same anchor, followed by six fixed epochs of PKU-MMD
  depth-only layer4 adaptation in three source-subject folds and an equal
  encoder-only source-fold soup for each seed.

Both arms used a bitwise-identical fresh 40-way head for each matching seed,
the same A–E target folds, temporal-difference weight 0.10, layer4+head scope,
optimizer, LR schedule, frozen BN, 15 fixed epochs and one held evaluation
after checkpoint reload. The runner verified that only encoder entries differed
before target training. All 15 jobs completed on GPUs 0 and 1 without opening
aggregate metrics early.

| Frozen endpoint | Control | PKU bridge | Delta / gate |
| --- | ---: | ---: | --- |
| Mean per-seed micro | 0.669521 | 0.677866 | +0.008344; passed ≥ +0.003 |
| Mean per-seed subject-macro | 0.666864 | 0.675394 | +0.008530; passed ≥ +0.003 |
| Per-seed micro delta (2026 / 2027 / 2028) | — | — | +0.008893 / +0.006917 / +0.009223; all passed |
| Per-seed subject-macro delta | — | — | +0.010067 / +0.005704 / +0.009819; all passed |
| Joint micro+subject non-degraded folds | — | — | 4 / 3 / 4; failed required ≥4 for every seed |
| Mean fold-cell worst-user delta | — | — | +0.015586; passed |
| Largest single-cell worst-user drop | — | — | −0.025339; failed −0.02 limit |
| Candidate micro / subject seed std | — | 0.004456 / 0.005086 | both passed ≤0.01 |
| Mean train-minus-held gap increase | — | — | −0.004006; passed ≤0.01 |
| Three-seed logit mean (diagnostic, not the gate) | 0.676877 | 0.685441 | +0.008564; worst-user 0.43125 → 0.45 |

This is a useful distinction: adding constrained PKU-MMD information produced
a real and consistent *average* target gain, including positive micro and
subject-macro deltas for every seed. The data hypothesis succeeded. Promotion
still failed because the preregistered gate required local fold robustness as
well as a higher mean: seed 2027 regressed on D/E, and its C worst-user drop
was 2.5339 pp. In total 8/10 criteria passed. The threshold was not relaxed
after observing the result.

An independent recomputation from all 15 candidate/control raw-logit pairs
matched every aggregate exactly. Frozen identifiers are: source materialization
`4110434a…153f`, preregistration `a4dc86f0…aa00`, runner
`49fcf679…dc14`, auditor `9cd7f284…60ba`, summary
`682223ca…3313`, decision `71625074…8243`, and seed-mean OOF logits
`741cfd38…1a02`. No anonymous/test asset, submission or leaderboard feedback
was read. The failed conjunctive gate therefore stopped full-data training,
fusion, packaging and submission, while preserving the positive external-data
finding for the next genuinely different preregistered method.

### 5.8 One-shot INT4 PKU deployment diagnostic, 2026-09-04 (completed; tied)

After §5.7, the user explicitly authorized one separate leaderboard diagnostic
for a sufficiently trained NTU/PKU model even if target OOF fluctuated slightly.
This did not reopen v4's failed thresholds. A new preregistration fixed one
seed-2028 checkpoint, 15 all-train epochs, layer4 + fresh-head scope,
temporal-difference weight 0.10, horizontal-flip TTA, and a 0.90 strictV3 /
0.10 bridge probability blend. Selection was train-only; the earlier v1
anonymous outputs were explicitly forbidden as v2/v3 design inputs.

Deployment precision, not GPU memory, was the limiting factor. The full FP16
bridge scored 2,059/3,036 OOF (0.678195) but could not fit beside strictV3 and
YOLO. Uniform INT3 collapsed to 364 rows; two preregistered mixed INT3/INT4
repairs collapsed to 355 and 365 rows. Neither mixed repair opened anonymous
test data. A train-only INT4 probe retained useful signal, so v3 froze exactly
one uniform per-output-channel signed INT4 candidate.

| v3 frozen endpoint | Result | Gate |
| --- | ---: | --- |
| INT4 external OOF | 1,930/3,036 = 0.635705; subject-macro 0.633421 | passed ≥1,800 rows |
| 90/10 probability blend OOF | 2,900/3,036 = 0.955204; worst-user 0.8125 | passed ≥2,898 rows and worst-user ≥0.8125 |
| Changed OOF top-1 vs canonical | 11 | passed ≥1 |
| Single checkpoint including YOLO | 99,978,253 bytes | passed <100,000,000 |
| Anonymous replay | two package-backed GPU logits arrays and two CSVs byte-identical; one changed prediction (index 36) | passed |
| Kaggle confirmation | **0.97512**, ref `56006027` | exact tie with canonical strictV3 |

The size repair did not quantize or otherwise alter strictV3. It removed only
the stored MobileNet thermal branch whose release weight is exactly zero, plus
the redundant top-level legacy `blend` and `yolo_bytes` records. Fusion,
visual, DSTFormer, temporal residual, and the executable release contract were
compared recursively and remained byte-equal after a safe `weights_only=True`
reload. The final package used legacy pickle protocol 2 because a pre-OOF
compatibility self-test showed that PyTorch 2.6's weights-only loader rejects
protocol 4 opcode 149; this correction was recorded before formal OOF.

The independent audit recomputed the 90/10 probabilities and submission,
verified all 63,464,372 quantized weights were INT4, and matched the frozen
package hash. Submission quota was checked at 5 remaining before upload and 4
afterward. The public score was first observed only after submission and was
not used to alter the candidate. Frozen identifiers are: preregistration
`44661e56…c701`, runner `c93c311b…b10c`, OOF summary `4a85db19…a60e`,
decision `da23595b…fd76`, package `054c9e46…c750`, auditor
`08780802…b3d`, audit record `8a671464…12b3`, and submission CSV
`b52ebd13…1869`.

The result supports the user's hypothesis in a bounded sense: the larger-data
initialization survived INT4 deployment and did not reduce public accuracy.
It did not improve accuracy, however. Because only one anonymous prediction
changed and the score tied, strictV3 remains the simpler canonical release;
the v3 bridge is preserved as report evidence rather than shipped on main.

### 5.9 NTU scale and coverage study (completed; no promotion)

The scale study compared fixed Kinetics-anchored NTU initializations under the
same seed-2026, 5-fold, 15-epoch layer4+head target protocol. Kinetics+25%
NTU120 reached 1,998 FP16 and 1,859 INT4 external OOF rows; its frozen 90/10
strictV3 blend retained 2,903/3,036, the 0.8125 worst-user floor, and changed
10 OOF decisions. The full fit ended at 0.962121 training accuracy. Its
99,971,501-byte single checkpoint produced two identical A100 replays and changed test
index 133 from class 24 to 19. Kaggle ref `56014518` scored **0.97512**.

An equal NTU60/NTU120 source update improved INT4 external OOF to 1,910 and
changed 14 blended OOF predictions while retaining 2,903 correct. Its final
CSV was nevertheless byte-identical to the prior PKU diagnostic (`b52ebd13…`),
so the novelty gate prevented a duplicate upload. Averaging the two target
models in weight space collapsed INT4 external OOF to 1,553 and failed before
test access. Finally, the deterministically retrained NTU60 control reproduced
2,044 FP16 correct rows but fell to 1,875 under INT4, below its frozen 1,900
gate; it also stopped before full-data training and test inference.

The evidence rejects the simplistic claim that more source samples must
monotonically improve this constrained deployment. NTU120 adds action coverage,
NTU60 remains more target-relevant in FP16, and the mixed source is more robust
to INT4 than NTU120 alone. None exceeded canonical strictV3, so the main
release remains unchanged. Three daily submissions were deliberately left
unused because no further candidate passed both the evidence and novelty
gates.

### 5.10 Fast close-out: packaging rule and NTU modality inventory

The 2026-09-04 close-out checked the two remaining assumptions without opening
another target fold. The organizer's ensemble ruling requires one checkpoint,
not merely a sum of separately supplied files. Actual single-file
materialisations gave:

| Inference artifact | Single-checkpoint bytes | Margin below 100,000,000 |
| --- | ---: | ---: |
| canonical strictV3 | 69,805,793 | 30,194,207 |
| PKU INT4 diagnostic | 99,978,253 | 21,747 |
| NTU120 INT4 diagnostic | 99,971,501 | 28,499 |
| NTU60/120 INT4 diagnostic | 99,975,005 | 24,995 |

All four pass, but the experimental artifacts have only 21–28 kB of genuine
headroom. Future packages are therefore gated on their final serialized file,
never on component-size arithmetic. The public strictV3 verifier now requires
the 69,805,793-byte bundle and validates its embedded detector hash.

The [provider inventory](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
states that NTU RGB+D 120 contains 114,480 samples and supplies masked depth,
skeleton and IR modalities; masked depth totals 147 GB and IR totals 389 GB.
Authenticated endpoint sizes matched all 32 local masked-depth archives, and
both skeleton archives are present. IR was incomplete at 9/32 setups
(97,333,511,077 complete-archive bytes); the missing 23 official archives total
320,783,737,075 bytes. A resumable direct download was tested, then stopped
after the resource/benefit review; no downloader remains active. The local
PKU-MMD Phase 2 tree was separately checked and contains complete three-view
depth frames but no usable IR. Consequently this report closes external IR
rather than training on an incomplete subset. RGB and full depth were never
downloaded for this question.

### 5.11 Quick non-IR tricks, 2026-09-04 (completed)

Two bounded screens were run. First, majority voting over strictV3,
`sched30_consensus` and `temporal_pool_consensus` recovered six strictV3 errors
without regression, but reached only 2,909/3,036 and therefore underperformed
the already frozen `sched30` result of 2,916/3,036. Confidence, margin, entropy
and class-prior nested routing likewise did not improve the fixed consensus.

Second, a train-only subject-balanced sampling screen targeted cross-user
robustness without changing inference weights. An initial comparison was
voided before reporting because its historical control used a different loader
worker stream. The replacement paired run fixed seed 2026, workers=4,
deterministic kernels, one head-only epoch, preprocessing contracts and all
other arguments; only `WeightedRandomSampler` changed. Fold A–E deltas were
−0.279, −0.892, −0.295, +1.636 and −1.461 pp. Mean accuracy moved from
89.1222% to 88.8641% (−0.2582 pp), only 1/5 folds was non-degrading, and the
worst fold fell by 1.4614 pp. It was rejected without test access or a second
sampler setting. The fixed equal-probability `sched30` consensus was the only
new small trick supported by the offline gates, but its 0.97014 public score
failed to support promotion.

### 5.12 Branch redundancy and native-rate cross-attention, 2026-09-05

The 15 restored strictV3 branch arrays first passed their recorded SHA-256
contracts. Their subject-wise OOF results show that Temporal is the primary
model, Visual is useful diversity, and Fusion is the weakest independent
encoder:

| Branch | Correct / 3,036 | Accuracy | Subject-macro | Worst user | Correct when both other branches are wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fusion | 2,430 | 0.800395 | 0.795598 | 0.478528 | 8 |
| Visual | 2,723 | 0.896904 | 0.900112 | 0.618750 | 30 |
| Temporal | 2,854 | 0.940053 | 0.939254 | 0.812500 | 61 |

A leave-one-branch-out diagnostic, retaining the already frozen per-fold
temperatures and gates, reached 2,909 rows for Temporal+Visual, 2,867 for
Temporal+Fusion and 2,766 for Visual+Fusion, versus 2,903 for all three. This is
post-hoc structural evidence rather than a promotable OOF candidate, but it
supports removing the duplicate Fusion R(2+1)D encoder from future models.

A new cache then retained every ordered native skeleton frame up to a padded
limit of 256. The median/p95/maximum were 23/69/236 frames; no clip was
truncated, no frame was interpolated or repeated, and 85,879 valid skeleton
frames were retained. One R(2+1)D-34 pass supplied eight layer-2 visual tokens.
A 135,465-parameter head encoded pose plus adjacent-frame velocity with
stride-1 convolutions, used skeleton frames as queries in four-head
cross-attention over the visual tokens, and added a zero-initialised residual
to the frozen baseline.

The fixed 12-epoch v1 moved Temporal from 2,854 to 2,844 rows (18 recoveries,
28 regressions). A single preregistered conservative follow-up used the
Fusion-free Temporal+Visual baseline, residual scale 0.25 and teacher-KL 1.0;
it moved 2,909 to 2,904 rows, with fold deltas 0/−1/−3/0/−1. Both failed. No
test inference was run and no scale/KL sweep followed. The architecture result
is therefore narrower than the design hypothesis: removing Fusion is supported,
but these high-rate cross-attention training recipes are not.

### 5.13 Independent public Depth/IR branches, 2026-09-05

A follow-up replaced the old Visual/Fusion encoders with independently
fine-tuned public Omnivore Swin-T branches while retaining the native-rate
skeleton encoder. No historical project checkpoint was loaded. On the fixed
716-row subject holdout, standalone Depth, IR and skeleton reached 0.472067,
0.544693 and 0.455307; their worst-user accuracies were 0.372549, 0.418301 and
0.359477.

Cross-attention did not improve this evidence. Two clip-level sensor tokens
reached 0.539106/0.411765 (accuracy/worst-user). Preserving eight time tokens
per sensor reached only 0.515363/0.392157 despite 0.9651 training accuracy, so
the learned fusion head was rejected as over-fit. A fixed equal-logit mean of
the three independently supervised branches reached 0.597765/0.503268. Because
the same held fold had already been inspected, that number is explicitly a
selection-biased complementarity diagnostic, not promotable OOF.

INT4 was unnecessarily conservative for these components. Per-output-channel
int8 backbone weights with sensitive patch embeddings, relative-position
biases, heads, norms and biases in fp16 serialized the two sensor models to
57,007,698 bytes total. Including the retained skeleton/temporal/detector
estimate gives 67,657,705 bytes before final bundle overhead. Thus int8 is the
preferred next deployment precision; int6 is only a size-gate fallback. At
this preliminary stage no anonymous-test inference was performed. In matched
batch-16 inference, mixed int8 changed Depth from 0.472067 to 0.467877 and kept
aggregate IR unchanged at 0.544693.

After explicit authorization, all three branches were retrained on all 3,036
rows with the same augmentation and fixed equal-logit rule. Unaugmented
in-sample accuracy was 0.950264 in fp32 and 0.950593 with the actual mixed-int8
bundle. The single bundle was 57,925,224 bytes. Its 405 test predictions covered
39 classes naturally (class 25 absent); none was edited to manufacture coverage.
Kaggle submission 56035438 scored 0.58706 publicly, consistent with the weak
held-fold evidence and far below strictV3. The route was rejected immediately;
the leaderboard result was not used for reweighting or another submission.

### 5.14 strictV3 Temporal anchor plus public Depth/IR, 2026-09-05

The next experiment restored strictV3 Temporal as the dominant branch and
trained the missing B–E Depth/IR folds, producing complete public-sensor OOF.
Depth and IR reached 0.492754 and 0.544466 versus Temporal's 0.940053. The
single frozen probability rule, Temporal/Depth/IR=`0.90/0.05/0.05`, reached
0.939723: fold deltas were −1/0/+1/0/−1 and worst-user remained 0.8125. It
therefore failed the all-fold non-degradation gate.

The sensors jointly corrected 23 of 182 Temporal errors, but agreed on the same
wrong class for 428 Temporal-correct rows. A separate nested weight diagnostic
reached 0.939065 and selected zero sensor weight for outer folds B, C and D.
The interface is retained with a fail-closed sensor gate of zero, exactly
recovering Temporal; no test inference or Kaggle submission followed.

### 5.15 Unified multi-stage Omnivore Depth/IR Fusion, 2026-09-05

To test whether late-logit fusion was discarding useful cross-modal structure,
two independently initialized Omnivore Swin-T trunks exposed all four stages
(`192/384/768/768` channels). Spatial pooling retained eight temporal tokens
per stage. Depth and IR then exchanged context through bidirectional four-head
cross-attention at every stage; the four summaries fed one unified Fusion
classifier, with final-stage modality heads used only for auxiliary supervision.

The fixed Fold-A screen used batch size 16, seed 2026 and 15 epochs. Augmented
training accuracy reached 0.816810, but held accuracy was 0.512570 and
worst-user accuracy was 0.431373. The auxiliary Depth and IR heads reached
0.407821 and 0.502793. Although the unified head improved on both auxiliary
heads, it remained below independently trained IR (0.544693), clip-token fusion
(0.539106), and the original strictV3 Fusion branch (0.800395 full OOF). This is
a cross-user generalisation failure; the fixed gate stopped B–E training,
anonymous-test inference and Kaggle submission.

### 5.16 Native single-trunk Omnivore RGB-D token fusion, 2026-09-05

A single Omnivore Swin-T then received `[IR, IR, IR, inverse-depth]`, explicitly
activating its four-channel `summed_rgb_d_tokens` implementation. Appearance
and depth use separate patch embeddings and are summed before the shared
transformer. Mean and standard deviation were fit only on the 2,320 outer-train
rows; initialization used only the official public checkpoint.

The fixed batch-16, 15-epoch Fold-A run reached 0.662069 augmented training
accuracy, 0.379888 held accuracy and 0.290640 worst-user accuracy. Although the
native code path was verified directly, performance fell below independent IR
(0.544693) and the dual-trunk multi-stage model (0.512570). The likely failure
is a pretraining-contract mismatch: Omnivore learned natural RGB plus metric
depth, whereas this dataset supplies repeated IR plus JET-inverted pseudo-depth.
The gate stopped B–E training, anonymous-test inference and Kaggle submission.

### 5.17 strictV3 Temporal + Visual exact equal fusion, 2026-09-05

This ablation removed legacy Fusion and disabled the per-sample quality gate.
After the frozen fold/full temperatures, Temporal and Visual logits received
exact `0.50/0.50` weights. No model was retrained; the unchanged 69,805,793-byte
strictV3 bundle contains every inference weight and remains below 100 MB.

Subject-wise OOF fell from strictV3's 2,903/3,036 (0.956192) to 2,881/3,036
(0.948946); fold accuracies A–E were 0.959497/0.936107/0.979381/0.955010/
0.901879. The 405 test predictions naturally covered all 40 classes and changed
15 strictV3 rows. The explicitly authorized Kaggle submission scored 0.95522
(ref 56036875), below strictV3's 0.97512. The ablation is rejected and the
release contract remains unchanged.

### 5.18 Temporal-dominant Visual optimization, 2026-09-05

The successful follow-up made only one structural change: legacy Fusion's base
weight was set to zero. All frozen strictV3 fold/full temperatures,
Temporal/Visual weights and confidence-quality gating were inherited. A
separate leave-one-fold-out weight search reached only 2,905 rows and regressed
folds A/E, so it was rejected rather than used to tune this candidate.

The inherited-gate candidate improved subject-wise OOF from 2,903 to
2,909/3,036 (0.958169). Fold deltas were 0/0/+5/+1/0, macro recall improved
from 0.953136 to 0.956076, subject-macro accuracy from 0.955794 to 0.957954,
and worst-user remained 0.8125. Confidence gating kept Temporal dominant: the
mean effective Visual weight on test was 0.1653. Predictions covered all 40
classes and changed only 3/405 strictV3 rows.

The authorized submission tied the canonical public score at 0.97512 (ref
56036959). It is retained as a simpler private-LB candidate, but the public tie
is not treated as evidence of superiority and no leaderboard-driven weight
search follows. The submitted artifact reused the compliant 69,805,793-byte
bundle; pruning the now-unused Fusion weights remains a release-packaging task.

A later deployment audit qualifies the 2,909 OOF figure: that diagnostic did
not apply the package-only `visual_package_output_scale=0.5`, whereas the
submitted compact-package inference did. The 0.97512 submission remains valid,
but 2,909 is not a deployment-matched OOF estimate.

### 5.19 Ordered Visual architecture priorities, 2026-09-07

Three cumulative Visual changes were trained from the same public IG-65M to
Kinetics-400 R(2+1)D-34 initialization, without loading any project checkpoint.
The protocol used the train-only Fold-A preprocessing contract, 2,320 training
rows, 716 held-subject rows, fixed 15 epochs and a single final held evaluation.

The control scored 448/716 (0.625698). A zero-initialized layer2/3/4 temporal
residual head fell to 436/716 (0.608939). Preserving layer4 temporal resolution
recovered seven rows to 443/716 (0.618715), but remained five below control.
A 114-parameter, zero-initialized bounded Depth/IR gate then reached 442/716
(0.617318). Worst-user accuracy was 0.517241 for control and 0.497537 for every
modified arm. The high-resolution result is directional evidence that late
temporal downsampling can matter, but no cumulative candidate passed Fold A;
B-E, full fit, anonymous-test inference and Kaggle submission were stopped.

### 5.20 Fusion-free quantization budget, 2026-09-07

Physical removal of Fusion and the inactive thermal placeholder reduced the
single checkpoint to 50,730,599 bytes while preserving the T/V temperatures,
weights, quality gate and Visual output scale. Its 405 predictions exactly
matched the zero-Fusion-weight candidate. Fresh quantization from original
FP16 fold weights showed no main-Visual benefit: 5/6/8-bit final T+V correct
counts were 2,893/2,892/2,893. A traceable NTU120 proxy fell from 1,788 FP16 to
1,730 INT4, while uniform 5-bit reached 1,804. Late-only mixed precision was
worse at 1,716, showing that its INT4 error is distributed through the trunk.
The attachment's separate 1,998/1,859 source checkpoint is unavailable, so it
was not falsely reconstructed from INT4.

Under the exact deployed 0.5 Visual scale, the regenerated 5-bit T+V OOF is
2,893/3,036; this is the appropriate quantization anchor rather than the
historical unscaled 2,909 diagnostic.

### 5.21 Temporal-conditioned Visual corrector, 2026-09-08

A small head now implements `z_new = z0 + r(hV, zT)`. Its hidden projection is
normally initialized and only the final layer is zero, making the initial
output exactly equal to `z0`. The fixed objective combines final CE with
`KL(p0 || p_new)` on correct, confidence-at-least-0.8 training rows.

Strict evaluation used frozen external-only encoders (PKU-MMD R(2+1)D-34
Visual and NTU DSTFormer skeleton Temporal), fresh fixed-final 40-class heads,
20 outer-by-inner cross-fitted runs, and five separate outer-train heads. Every
upstream target head excluded its outer-held users. No anonymous-test input was
accessed. The 3,036-row baseline improved from 1,201 (0.395586) to 1,465
(0.482543): 438 corrected, 174 broken, net +264. Fold A-E net changes were
+46/+58/+48/+66/+46 and all 18 users improved. Subject-macro accuracy rose from
0.391962 to 0.479934, worst-user from 0.1625 to 0.29375, and the user-clustered
95% accuracy-delta interval was [+0.07310,+0.10375]. A second run reproduced
both OOF arrays byte-for-byte.

This confirms the correction mechanism under strict isolation, not its gain on
deployed strictV3. The external-only proxy is substantially weaker; Visual
scored 1,310 while Temporal scored 902, making frozen strictV3
Temporal-dominant fusion weights mismatched. The candidate still exceeded
Visual alone by 155 rows, but the result does not authorize full-fit, anonymous
test inference, or Kaggle submission.

Deployment-aligned validation then regenerated each held-user Visual output by
freshly applying uniform 5-bit quantization to the original FP16 fold weights,
with the exact release single-view preprocessing and 0.5 package scale. The
resulting T+V baseline was 2,890/3,036 (0.951910), and its test predictions
matched the already submitted T+V CSV exactly. A corrector using the 512-D
pre-classifier features lost 24 rows. The single preregistered alignment change,
replacing checkpoint-specific features with 40-D class logits, still lost 11
rows: 32 corrected versus 43 broken, with fold nets +6/+9/+6/0/-32.
Subject-macro accuracy fell from 0.951297 to 0.947341 and worst-user accuracy
from 0.8125 to 0.625.

The error is a concrete generalization bias: 30 correct class-36 examples from
user 5 were redirected to class 10, accounting for most of the Fold-E failure.
Both preregistered deployment screens failed the non-degradation gates.
Consequently the corrector was neither full-fit nor submitted to Kaggle; doing
so would turn a failed confirmatory endpoint into leaderboard-guided tuning.

### 5.22 Input validity and action-invariance controls, 2026-09-08

A source-decoding audit, rather than a black-pixel heuristic, found 2,933 clips
with usable Depth-or-IR and 2,931 with at least one parsed skeleton frame. There
are 103 clips with neither primary input. The frozen release correctly predicts
only 30/103, so these clips contribute 73 of its 133 OOF errors; 60 belong to
user 5 and 30 to class 36. This confirms a severe missingness confound and its
overlap with the corrector's earlier class-by-user failure.

Three preregistered, fixed-final subject-OOF controls were then tested without
anonymous-test access. First, every row remained in the forward pass and
BatchNorm stream, while CE gradients were removed only for branch-invalid rows.
The paired frozen-gate T+V control scored 2,889; Visual-only, Temporal-only and
both-branch masking scored 2,888/2,885/2,885. Both masking corrected 8 and broke
12 rows (net -4), with folds -1/-5/+2/0/0. The valid-input subset and
subject-macro accuracy also fell.

Second, a naive fixed-absolute-length skeleton reconstruction was rejected by
geometry QC before training because maximum normalized displacement reached
2.474. The accepted variant used clip-fixed bilateral ±5% bone multipliers and
preserved projected motion, direction, root, confidence, and missing frames;
mean/max displacement was 0.0153/0.1460. It changed Temporal 2,847→2,846 and
T+V 2,889→2,888, with no corrected row and one broken row.

Third, a weight-0.05, temperature-0.1 supervised contrastive term acted on the
40-D representation used by the TCN classifier. Positives were same-class,
different-user outer-train clips; different classes were negatives and
same-class/same-user pairs were ignored. Every fold had at least 2,118 valid
anchors per epoch and decreasing contrast loss, but Temporal changed
2,847→2,843 and T+V 2,889→2,888 (5 corrected, 6 broken; fold nets
-1/-1/+1/0/0). Thus signal availability was not the issue. All three candidates
failed their gates, and no post-result mask, magnitude, weight, or temperature
scan was performed.

### 5.23 Temporal modelling before the 40-class frame head, 2026-09-09

Before retraining, the three §5.22 treatments were decomposed by source-input
validity. None corrected or broke any of the 103 rows with neither primary
input. All changes occurred among the 2,931 rows with both primary inputs,
confirming that the negative results reflected action-discrimination changes
rather than a changed missing-input default.

A preregistered paired experiment then isolated whether the final 40-class
frame head discards motion evidence. One released DSTFormer forward pass
exported both frame logits and its post-ReLU 2,048-D representation immediately
before `fc2`; the logits exactly matched the prior control cache (maximum
difference zero and identical SHA256). For each outer fold, PCA-40 was fitted
only on frames from outer-training users and frozen. Its explained-variance
sum was 0.655–0.670. The static path remained the mean of the same 16 frame
logits, while only the equal-capacity TCN input changed from logits to PCA
features. Architecture, seed, training budget, Visual outputs and release T+V
gate were unchanged.

Feature-TCN reduced Temporal from 2,847 to 2,839 correct. After frozen T+V
fusion it changed 2,889 to 2,891: six corrected and four broken, with fold nets
0/-1/+2/0/+1. Both gains occurred on valid-primary rows; the 103 missing rows
were unchanged. Subject-macro accuracy rose by 0.000777 and the preregistered
minimal gate technically passed. However, exact two-sided McNemar p was 0.7539,
the user-cluster bootstrap 95% interval for accuracy delta was
[-0.000983, 0.002628], and Temporal alone degraded in three folds. Corrections
and failures also swapped within the same confusable actions and users. This is
weak fusion-interaction evidence, not confirmation that the 40-D head is the
main temporal bottleneck. The candidate is not promoted, full-fit, submitted,
or followed by PCA-width/attention scans. A local-view experiment remains
conditional on direct original-frame evidence of crop, resolution, or sampling
loss.

### 5.24 Current T+V error coverage and raw-to-cache evidence, 2026-09-09

The error list was reset to the current paired T+V control, not inherited from
the older 2,903-row release. Its 2,889/3,036 OOF result has 147 errors: 73 among
the 103 no-primary clips and 74 elsewhere (72 with both primary inputs and two
with only one). Among those 74, Temporal alone has the correct top-1 for six,
Visual alone for 53, and both top-1 predictions are wrong for only 15. For all
15 joint errors, Temporal ranks the true class second; Visual ranks it second
in 11, third in one, and 13/25/36 in the remaining three. Thus most current
valid-input failures concern preservation of evidence already present in one
branch. This is neither a soft-fusion upper bound nor permission to tune branch
weights on these OOF errors.

Six directly comparable completed candidates were then used only as review
priority markers. They ever corrected 10 unique baseline errors; 137 persisted.
The persistent set contains all 73 no-primary errors, six Temporal-only-correct,
45 Visual-only-correct and 13 joint-top-1 errors. Candidate outputs were not
combined and these rows were not converted into a training target.

For input evidence, one row from each of ten independent persistent joint-error
clusters was paired with a same-class, different-user correct control. Before
opening labels, predictions or error/control roles, the reviewer compared every
original Depth_Color frame, the actual 16 uncropped selections, the exact
release-contract union crop resized to 128×128 in Depth and IR, and the matched
H36M-17 skeleton. No clear sampling-stage or spatial-processing loss appeared
in either group. Suspected skeleton anomalies occurred in 2/10 errors and 2/10
controls; Depth, IR and skeleton source-frame counts aligned in all 20 clips.
Median maximum normalized skeleton step was 0.488 for errors and 0.629 for
controls. This limited descriptive sample does not prove absence of recoverable
raw evidence, but it exposes no repeated cross-user mechanism that authorizes
sampling, local-view, resolution, or skeleton-repair training.

---

## 6. General method-discipline rules (hard constraints)

1. **Never** use test / anonymous labels, prediction history,
   submission score or leaderboard row-level feedback as design
   signals. Any candidate that touches them is voided.
2. **Outer A–E / inner C-E is a confirmatory endpoint**, not
   feedback. Once revealed on a candidate, no further hyperparameter
   scans on the same direction are admissible.
3. Every new candidate must be accompanied by: a new preregistration,
   a new formal route, a new independent audit, P0/P1 closed. We do
   not accept "existing candidate + change one line".
4. **Mandatory gate sequence**:
   * full-data OOF nested-CV (5/5 fold non-degrading, subject-macro
     same direction, worst-user improved, cross-seed stable);
   * two-round raw replay (CSV byte-equal, logits `max_abs` ≈ 0,
     bit-width / precision match between deployment and training OOF);
   * natural 40-class coverage;
   * one checkpoint containing every inference weight < 100,000,000 bytes;
   * independent audit SHA consistent;
   * only then write `submit_candidate.zsh` (and only when the Kaggle
     quota allows).
5. **No relaxations**: an independent audit that finds a real P0/P1
   voids the candidate. We do not relax the package budget, the
   gate, or the audit to make a candidate fit.

---

## 7. The next concrete work list (ordered)

1. Keep the byte-replayed `sched30_consensus` as frozen negative deployment
   evidence after its 0.97014 public result; do not retune or resubmit it.
2. **CPU materialisation of the outer-train-only normalisation
   builder + matched CV runner** for the from-scratch TSM/S3D path
   (reads only labelled-train cache + metadata; never opens
   held/test/anonymous/submission). Run 80 inner + 30 outer synthetic
   regression.
3. **Keep PKU/NTU external IR closed.** Preserve the completed depth-only PKU
   and NTU evidence, but do not train or report an incomplete paired-IR model.
   Also keep the PKU deployment line closed after the §5.8 public tie; do not
   retune bridge weight, precision, epoch, seed, or the single changed sample
   from leaderboard feedback.
4. Any new mechanism candidate is mirrored into
   `BEST_REPORT_EVIDENCE_MANIFEST_*.json` (with SHA + decision) and
   into `EXTERNAL_ONLY_RESEARCH_ROADMAP_20260830.md` (the prior
   becomes "authorized" or "rejected").
5. Report / evidence close-out: every executed step above writes its
   result into the relevant report's *fact-freeze-date* paragraph
   with the new manifest SHA.

---

## 8. Reproducing and auditing strictV3

The artifact-level check reproduces the reported OOF score and canonical CSV.
The raw-data replay reproduces the scored 0.97512 CSV byte-for-byte:

```zsh
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.replay \
  --output results/strict_v3/reproduced_submission.csv
.conda/envs/cuhkx/bin/python -m yolo_r2plus1d.strict_v3.release.verify \
  --replayed-csv results/strict_v3/reproduced_submission.csv
```

Any replay other than the exact canonical hash fails closed.

## 9. Compliance and rules summary

The project never uses test/anonymous labels, IDs, samples, submission scores,
or post-hoc leaderboard feedback as design signals. The September replay was
submitted only as a confirmatory reproduction audit; its score was not used to
tune either differing prediction. The §5.8 external-data diagnostic was also
fully frozen before anonymous inference; its tied score was recorded and the
route closed without retuning. The historical compliance chain is summarised
inline in §4.6.

## 10. Evidence anchors (file paths under this repo)

* [`checkpoints/strict_v3/model.pt`](../checkpoints/strict_v3/model.pt) and
  [`yolo11n.pt`](../checkpoints/strict_v3/yolo11n.pt) — deployable model.
* [`results/strict_v3/release_manifest.json`](../results/strict_v3/release_manifest.json)
  and [`metrics.json`](../results/strict_v3/metrics.json) — release contract and
  OOF evidence.
* [`results/strict_v3/submission.csv`](../results/strict_v3/submission.csv) —
  scored 0.97512 canonical CSV.
* [`results/strict_v3/raw_replay/`](../results/strict_v3/raw_replay/) —
  byte-identical raw-data replay.
* [`verify.py`](../yolo_r2plus1d/strict_v3/release/verify.py) and
  [`replay.py`](../yolo_r2plus1d/strict_v3/release/replay.py) — public
  verification and end-to-end replay entry points.
