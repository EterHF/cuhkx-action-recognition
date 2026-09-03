# CUHK-X Small Model Track — Technical Report

[English](TECHNICAL_REPORT.md) | [简体中文](TECHNICAL_REPORT.zh-CN.md) | [Documentation index](README.md) | [Repository home](../README.md)

**Best public Kaggle score so far: `0.97512` (rank 3, ref `55712568`)**<br>
**Gap to rank 2: `0.00497` (one public sample's net swing).**<br>
**Frozen: 2026-09-03**

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

- 8 ordered frames of **Depth (Color / IR)** and **Thermal** (aspect-ratio
  preserving padding).
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
| Package | `checkpoints/strict_v3/model.pt` (64,206,000 bytes) |
| Canonical submission | `results/strict_v3/submission.csv` (405 rows, 40 classes) |
| Raw-replay submission | `results/strict_v3/raw_replay/submission.csv` (405 rows, 40 classes) |
| Total with YOLO11n | 69,819,764 bytes (under 100 MB) |
| Public score | **0.97512**, ref `55712568` |
| Reproduced raw-data score | **0.97512**, byte-identical to ref `55712568` |
| OOF (release-strict) | aggregate `0.956192`, mean fold `0.954823`, worst fold `0.926931`, macro recall `0.953143` |
| Protocol | `five_fold_subject_wise_nested_temperature_quality_gate` (full contract in `release_manifest.json`) |
| Raw-replay evidence | deterministic SHA-256 `e2509491…`; byte-identical to the canonical CSV |

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

Multiple offline "higher OOF" candidates (Fusion4 raw at 0.970*, etc.)
were **refused by the public leaderboard**; they are no longer candidates.

### 3.2 How to reproduce `legal_strict_v3`

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

### 3.3 Retraining audit on the cleaned repository

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
these directions ended up *exhausted* (further work would only inflate
the validation-set search space).

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

### 4.3 External-data scaling (exhausted)

NTU Depth / Skeleton pretraining and PKU-MMD bridge were explored
heavily. The audit summary:

| Experiment | Result | Verdict |
| --- | ---: | --- |
| NTU masked-depth pretraining + R(2+1)D-18 | 62.98% (control 58.27%), 5/5 fold gain, worst-user +12.50 pp | Real positive-transfer signal at small backbone |
| NTU60 / NTU120 direct R(2+1)D-34 supervision | 58.56% / 58.89% | Large source-domain supervised training **forgets** the Kinetics representation |
| Kinetics-anchor + 25% NTU60 encoder interpolation | 65.18% | Keep Kinetics anchor, but did not improve final strict-v3 |
| NTU Skeleton student | source val 64.68% at epoch 25 | External skeleton encoder usable; still needs 5-fold target validation |
| PKU-MMD source-only pretraining | source val 75.30%, source train ~99% at epoch 15 | 6,952 records clearly over-fit; source validation does not proxy target transfer |
| Kinetics → PKU → CUHK-X | 61.92% (control 63.27%) | Direct PKU continued-training is **negative transfer** |
| PKU visual replacing strict-v3 visual | 95.191% (control 95.619%); nested mean weight = 0 | Cannot serve as a 4th logit branch |

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
* **PKU-MMD licence**: the public page does not grant redistribution
  rights for prize competitions; the V3 GPU handoff is *paused* until
  a written permission / clarification is obtained.
* **MViTv2-S**: official checkpoint ≈ 131.9 MB, exceeds the 100 MB
  budget. **Held in limbo** — pending written organiser classification.
* **TorchVision licence reminder**: pretrained weights may inherit
  training-data terms; documented in the External-Compliance review.

---

## 5. Methods and experiments still under consideration

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
  86.43 MB / 92.05 MB with YOLO, well under 100 MB).
* **Outstanding step**: deployment / training-OOF precision and
  bit-width match (FP16 ↔ FP32 DSTFormer mismatch is known).
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
    non-degrading, every seed 5/5, 92.05 MB / 92.05 MB with YOLO, CSV
    `f33e0569…`.
  * Temporal-pooling majority vote (top-2 + energy): OOF +0.003623, 5/5
    non-degrading, 88.43 MB / 96.96 MB with YOLO, CSV `6a320486…`.
* **Next-step design points**:
  1. Treat the two pooling variants as a structural minimum, and
     perform strict OOF nested selection — *not* a fixed 50/50 blend.
     No manual coefficient picking.
  2. New pooling variants are accepted only if they share the
     DSTFormer frame logits with the anchor / `sched30` and add no
     large weights.
  3. Every candidate must produce two byte-equal CSVs and a deployment
     logits that matches the training OOF exactly in FP16/FP32.

### 5.3 From-scratch small backbone — "competition-only" (priority 3, compliance fallback)

* **Motivation**: the Small Track ≤ 100 MB and "no large pretrained
  backbone" rules plus the PKU-MMD licence ambiguity and the
  MViTv2-S 131.9 MB checkpoint force a *different* method that
  trains from scratch.
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

### 5.4 External representation transfer (priority 4 — only when 5.1–5.3 fail)

The external-only research queue (verbatim from the prior roadmap):

1. **VideoMAE-S ≈ MViTv2-S** — same investigation tier; the "≈" means
   "same investigation tier", not "equivalent on this dataset".
   Freeze the external checkpoint / architecture / preprocessing / seed
   map before opening any CUHK-X fold.
2. **Deterministic depth / lag-1 temporal-difference channels** —
   fixed, auditable; pre-declared auxiliary weight. No held-driven
   channel-recipe search.
3. **Predeclared tail weight averaging (executed; rejected)** — the fixed
   epoch 3–5 interval, LR schedule, parameter scope and BN treatment were
   declared before A–E. The averaged checkpoint was the only candidate;
   results are frozen in §5.5 and no post-hoc SWA schedule scan is allowed.
4. **SlowFast / X3D** — same subject folds, seed budget, source-only
   data boundary; measure compute / accuracy locally first.
5. **MixStyle / ASAM** — last in queue, because small-fold spurious
   gains tend to damage cross-seed stability.

External-only is *strict*: external representation / public
architecture / deterministic training transform. **Never** the test /
anonymous labels / IDs / samples / submission scores / prediction
history / post-hoc leaderboard feedback.

### 5.5 Temporal generalisation checks, 2026-09-03 (completed; rejected)

Two one-variable candidates were locked before their A–E runs. Both used the
same train frame-logit SHA `76e6c11…7102e`, metadata SHA `bf2e93e5…9f099`,
seeds 2026/2027/2028, five fixed sched30 epochs and deferred held metrics.
The suite ran in OOF-only mode: 15/15 jobs completed, no full-data model or
test file was created, and every receipt recorded `test_data_loaded=false`.

| Candidate | Temporal result vs fresh sched30 | Generalisation gate | Release gate / decision |
| --- | --- | --- | --- |
| Epoch-varying deterministic reversal/noise | 2,847/3,036 = 0.937747; seed 2026 gained one row but the seed mean changed zero predictions | macro, subject-macro, worst-user, worst-fold and 5/5 folds tied; failed the preregistered +1-row minimum | Stopped at stage 1; rejected |
| Uniform FP32 TCN parameter average, post-update epochs 3–5 | 2,848/3,036 = 0.938076 (+1 row); macro +0.000327; subject-macro +0.000208; worst metrics tied | 4/5 folds and 3/3 matching seeds non-degrading; passed stage 1 | nested branch 2,920/3,036 = 0.961792 (+1 vs historical branch), but fixed 50/50 strictV3 consensus stayed 2,916/3,036 = 0.960474 with zero changed predictions; required 2,917, so rejected at stage 2 |

The train-only nested evaluator was first checked against the historical
sched30 inputs: both the saved nested logits and the final probability array
were exactly equal (`max_abs=0`). Evidence hashes are: epoch-resampled receipt
`dd135d65…df24`, OOF `0ddb3bdd…67c0`; tail-average receipt
`40510e90…26ac`, OOF `99a0df0b…5960`; final nested-gate metrics
`d8edb37a…044b2`. The stop rules prohibited full-data training, anonymous-test
inference, packaging and coefficient changes after seeing A–E. In accordance
with repository cleanup policy, the rejected mechanisms remain documented
here rather than as permanent training flags or model files.

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
   * package + YOLO ≤ 100 MB;
   * independent audit SHA consistent;
   * only then write `submit_candidate.zsh` (and only when the Kaggle
     quota allows).
5. **No relaxations**: an independent audit that finds a real P0/P1
   voids the candidate. We do not relax the package budget, the
   gate, or the audit to make a candidate fit.

---

## 7. The next concrete work list (ordered)

1. **Complete the nested / shared-state multi-pooling two-round raw
   replay** for `sched30` and `fp32_consensus`. Promote to the head
   of the submission queue once the gates pass.
2. **CPU materialisation of the outer-train-only normalisation
   builder + matched CV runner** for the from-scratch TSM/S3D path
   (reads only labelled-train cache + metadata; never opens
   held/test/anonymous/submission). Run 80 inner + 30 outer synthetic
   regression.
3. **Only when 1 + 2 still fail to crack the public top-2**, apply
   for a new preregistration in the priority-4 external queue.
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
tune either differing prediction. The historical compliance chain is
summarised inline in §4.6.

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
