# Optimization of Deep Learning Models for Edge Deployment

**Master's Thesis — Università degli Studi di Milano**
IEBI Lab, Department of Computer Science
Supervisor: Angelo Genovese
Target hardware: STM32N6570-DK (ST Neural ART NPU)
Task: Binary person detection (Visual Wake Words)

> **v2 — 11 June 2026.** Every number in this document is traced to a board log (`logs/`), an experiment record (`checkpoints/*.json`, `outputs/final_test_results.json`), or marked **pending**. Accuracies carry Wilson 95% CIs. Changes vs v1: corrected flash figures (now from on-board octoFlash, not ONNX weight payload), corrected NPU epoch counts, added hardware NSE for every deployed model, removed the untraceable MV2-baseline INT8 accuracy, added Phase 6 (physical rebuild), reframed the Phase 5 "anomaly" as an open question, corrected the quantization-granularity description (weights are per-channel).

---

## Phase 1 — Dataset

**Source:** Silicon Labs VWW (COCO2014-based, 96×96 pre-resized)
**Split:** Fixed 70/15/15 per class, seed 41, manifest files (verified balanced)

| Split | Samples | Per class | Purpose |
|---|---|---|---|
| Train | 7,000 | 3,500 | Model training; INT8 calibration (200 samples) |
| Validation | 1,500 | 750 | Checkpoint selection, early stopping |
| Test (held-out) | 1,500 | 750 | Final evaluation only |
| STM32 batch | 200 | 100 | First 100 per class of the (seed-shuffled) test manifests; identical file (`test_input.npz`) across all board runs |

**Augmentation (train only):** RandomResizedCrop(96, 0.7–1.0) · HFlip · Rotation ±15° · ColorJitter(0.2) · ImageNet normalize. Val/test: resize + normalize only.

**Statistical notes (apply to every table below).**
- n=1,500 → 95% CI width ≈ ±2.1 pp. Smallest unpaired-significant difference between two models: ≈ 2.9 pp.
- n=200 → 95% CI width ≈ ±5.5 pp. Smallest unpaired-significant difference: ≈ 8 pp.
- **No accuracy delta reported in this document reaches unpaired significance.** Paired McNemar tests on shared per-sample predictions are required and pending (Final_Results cell 7 already collects the predictions).

---

## Phase 2 — Baselines

Training: Adam · LR 1e-3 · batch 64 · CosineAnnealing · ≤50 epochs · patience 10 · label smoothing 0.1. Teachers: 3-phase unfreeze, ≤30 epochs.

| Model | Role | Params | Val % (n=1500) | Test % (n=1500) | Test 95% CI |
|---|---|---|---|---|---|
| MobileNetV2 | student | 151,874 | 78.40 | 79.47 | 77.4–81.4 |
| MobileNetV3 | student | 139,428 | 79.13 | 79.13 | 77.0–81.1 |
| VGG16-BN (pretrained) | teacher | 27,634,626 | 89.07 | 90.60 | 89.0–92.0 |
| ResNet50 (pretrained) | teacher | 24,623,042 | 87.93 | 89.73 | 88.1–91.2 |
| VGGStyle (scratch) | comparison | 5,958,242 | 80.27 | — | — |
| ResNet (scratch) | comparison | 2,828,194 | 77.93 | — | — |

### Hardware: FP32 vs INT8 on STM32N6 (from board logs, mode=TARGET)

| Model | Format | Latency/sample | Throughput | octoFlash (weights) | RAM (activations) | NPU epochs | HW NSE |
|---|---|---|---|---|---|---|---|
| MV2 | FP32 | 538.85 ms | 1.86 inf/s | 622.5 kB | 906.5 KiB | 2 / 47 HW | 1.000 |
| MV2 | INT8 | **2.833 ms** | **353 inf/s** | **233.9 kB** | **242.1 KiB** | 25 / 25 HW | **0.711** ⚠️ |
| MV3 | FP32 | 385.43 ms | 2.59 inf/s | 560.6 kB | 679.1 KiB | 4 / 96 HW | 1.000 |
| MV3 | INT8 | **3.659 ms** | **273 inf/s** | **169.3 kB** | **216.0 KiB** | 55 / 55 HW | 0.997 |

**Verified finding:** INT8 triggers full NPU offload. Latency drops **190×** (MV2) and **105×** (MV3). Flash drops **62.4%** (MV2) and **69.8%** (MV3); activation RAM drops **73.3%** and **68.2%**. (v1 reported flash from the ONNX weight payload and overstated the reduction as ~73%.)

**Board accuracy (200-sample batch):** MV3 FP32 79.5% → INT8 80.0% (CI 73.9–85.0). MV2 FP32 76.5% (CI 70.2–81.8); **MV2 INT8 board accuracy is unrecorded** — `quantz_records.json` holds `null`; the v1 figure of 76.0% has no traceable source. Re-extract from `output_int8.npz` or re-run validation before citing any MV2 INT8 accuracy.

⚠️ **MV2 INT8 hardware NSE is 0.711** — far below the 0.95 deployment gate this thesis applies elsewhere. On-device outputs diverge substantially from the reference INT8 model. Any board-accuracy claim built on this deployment is unreliable until the divergence is explained.

---

## Phase 3 — Knowledge distillation

Config: T=4.0 · α=0.7 (Hinton et al., 2015 defaults; no ablation was run) · ≤80 epochs · patience 20. 8 runs: 2 teachers × 2 students × {warm-start, scratch}.

| Run | Val % | Test % (CI, n=1500) | Δ val vs baseline | Local NSE | HW NSE | STM32 INT8 % (CI, n=200) |
|---|---|---|---|---|---|---|
| vgg_mv2_ft | 80.07 | 81.00 (78.9–82.9) | +1.67 | 0.987 | **0.397** ⚠️ | 80.0 (73.9–85.0) |
| vgg_mv2_scratch | 79.53 | 80.40 (78.3–82.3) | +1.13 | 0.983 | 0.999 | 79.0 (72.8–84.1) |
| resnet_mv2_ft | 79.47 | 80.67 (78.6–82.6) | +1.07 | 0.961 | **0.719** ⚠️ | 82.0 (76.1–86.7) |
| resnet_mv2_scratch | 79.53 | 81.33 (79.3–83.2) | +1.13 | 0.978 | 0.983 | 84.0 (78.3–88.4) |
| vgg_mv3_ft | 79.13 | — | 0.00 | 0.991 | not deployed | — |
| vgg_mv3_scratch | 79.53 | — | +0.40 | 0.992 | not deployed | — |
| resnet_mv3_ft | 79.13 | — | 0.00 | 0.993 | not deployed | — |
| resnet_mv3_scratch | 79.53 | — | +0.40 | 0.987 | not deployed | — |

**Honest reading.** Every MV2 KD run improves val and test point estimates by 1–2 pp; the direction is consistent across 4/4 runs, but no single delta is unpaired-significant (threshold ≈ 2.9 pp at n=1,500). Paired McNemar is required before claiming a KD gain. MV3 shows no KD response (0.00–0.40 pp).

⚠️ **Two of the four deployed KD models fail the hardware NSE gate** (vgg_mv2_ft 0.397, resnet_mv2_ft 0.719) — a fact absent from v1. Until the MV2-architecture INT8 hardware divergence is diagnosed, no board-accuracy comparison among MV2 deployments is defensible. Note the pattern: both *warm-started* MV2 KD models fail HW NSE; both *scratch* ones pass (0.999, 0.983).

---

## Phase 4 — Unstructured pruning (verified negative finding)

L1 magnitude, all Conv2d · 10-epoch FT (LR 1e-4) · all 6 checkpoints passed the local NSE gate and were deployed.

| Model | Target | After FT % (val) | Test % (CI, n=1500) | Local NSE | HW NSE | STM32 INT8 % (n=200) |
|---|---|---|---|---|---|---|
| MV2 | 10% | 78.93 | 80.40 (78.3–82.3) | 0.982 | 0.957 | 78.5 |
| MV2 | 20% | 78.53 | 79.73 (77.6–81.7) | 0.984 | 0.956 | 79.5 |
| MV2 | 30% | 78.27 | 78.47 (76.3–80.5) | 0.980 | 0.954 | 76.0 |
| MV3 | 10% | 78.67 | 79.13 (77.0–81.1) | 0.984 | 0.995 | 80.5 |
| MV3 | 20% | 78.27 | 79.07 (76.9–81.1) | 0.987 | 0.995 | 81.0 |
| MV3 | 30% | 78.47 | 79.07 (76.9–81.1) | 0.989 | 0.995 | 83.0 |

**Hardware metrics are byte-identical across all sparsity levels and the dense baseline** (MV2: 2.833 ms / 233.9 kB / 242.1 KiB / 25,663,842 MACC at every level; MV3: 3.66 ms / 169.3 kB / 216.0 KiB / 20,387,164 MACC). **The Neural ART NPU executes dense INT8 regardless of weight sparsity — unstructured pruning yields zero hardware benefit on this target.** This is the most strongly verified finding in the thesis.

---

## Phase 5 — Structured pruning (gate results verified; interpretation revised)

L2 filter-level, pointwise Conv2d only · 15-epoch FT.

| Model | Target | After FT % (val) | Local NSE | Gate (≥0.95) | Deployed |
|---|---|---|---|---|---|
| MV2 | 2% | 79.33 | 0.9324 | FAIL | no |
| MV2 | 3% | 78.73 | 0.9257 | FAIL | no |
| MV2 | 5% | 78.53 | 0.8471 | FAIL | no |
| MV3 | 2% | 78.87 | 0.9860 | pass | yes |
| MV3 | 3% | 78.73 | 0.9728 | pass | yes |
| MV3 | 5% | 78.33 | 0.8252 | FAIL | no |

Deployed: MV3 struct 2% → STM32 INT8 83.0% (CI 77.2–87.6), HW NSE 0.997; MV3 struct 3% → 84.5% (CI 78.8–88.9), HW NSE 0.996. Latency/flash/RAM unchanged vs baseline (masked filters remain dense at 1.8–2.9% sparsity).

**Revised interpretation.** v1 declared structured pruning + static INT8 "fundamentally incompatible," attributing INT8-above-FP32 accuracy to miscalibrated *per-tensor* quantization. Three corrections:
1. The comparison behind the "anomaly" (INT8 on the 200-batch vs FP32 on the 1,500 val set) is invalid — the pipeline itself labels the 200-sample figure "not comparable to FP32 val." FP32 accuracy *on the same 200 samples* was never measured for these models. The MV3 baseline already scores ~1 pp higher on the 200-batch than on val, and the 200-sample CI is ±5.5 pp; the "anomaly" may be batch composition plus noise.
2. The pipeline quantizes weights **per-channel** (`per_channel=True`); only activations are per-tensor. The v1 root-cause description mischaracterizes the pipeline. (Open: stedgeai ingests the model as "sa/sa per tensor" — confirm what Neural ART actually compiles.)
3. High HW NSE (0.996–0.997) confirms faithful deployment — whatever is happening is a property of the quantized model or the evaluation set, not the hardware.

**Status: open question, not a concluded incompatibility.** Decisive, cheap experiment: evaluate every FP32 ONNX on the 200-batch locally and compare like-for-like.

---

## Phase 6 — Physical rebuild of structured-pruned MV3 (new; supersedes part of Phase 5)

Channels physically removed (torch-pruning), 15-epoch FT (best-val checkpointing), INT8 export, board deployment (9 June).

| Model | Params | Δ params | MACC (stedgeai) | Latency | vs dense 3.659 ms | octoFlash | HW NSE | STM32 INT8 % (n=200) |
|---|---|---|---|---|---|---|---|---|
| MV3 dense baseline | 139,428 | — | 20,387,164 | 3.659 ms | 1.00× | 169.3 kB | 0.997 | 80.0 |
| MV3 rebuilt 2% | 132,662 | −4.85% | 19,273,690 | **9.216 ms** | **0.40× (2.5× slower)** | 140.8 kB | 0.996 | 80.5 |
| MV3 rebuilt 3% | 130,006 | −6.76% | 18,928,060 | **5.230 ms** | **0.70× (1.4× slower)** | 147.1 kB | 0.994 | 83.0 |

Post-FT test accuracy (n=1500): 78.07% (2%) and 79.60% (3%) — **provisional**: a workflow bug measured these on last-epoch weights while the deployed checkpoint holds best-val weights; re-evaluate the actual checkpoints. MAC-proportional latency prediction (≈3.4–3.5 ms) missed reality by 1.5–2.7×; note also rebuilt-3% uses *more* flash than rebuilt-2% despite fewer parameters — compiler memory layout dominates at this scale.

**New negative finding:** removing 5–7% of channels made inference **1.4–2.5× slower** despite fewer parameters and MACs. The Neural ART compiler's efficiency depends on hardware-aligned channel counts; small structured pruning breaks that alignment and the schedule degrades. On this target, structured pruning at low ratios is not merely useless — it is **counterproductive**.

---

## Key findings (v2)

| # | Finding | Status |
|---|---|---|
| 1 | INT8 quantization triggers full NPU offload: 105–190× latency reduction, 62–70% flash reduction, 68–73% activation-RAM reduction | ✅ Verified (logs) |
| 2 | Unstructured pruning yields zero hardware benefit — the NPU executes dense INT8 regardless of sparsity | ✅ Verified (logs, byte-identical metrics) |
| 3 | Physically rebuilt structured pruning is counterproductive: 1.4–2.5× *slower* despite −5–7% params/MACs (channel-alignment penalty) | ✅ Verified (logs); accuracy figures provisional |
| 4 | MV3 quantizes far more reliably than MV2 on this hardware: HW NSE 0.994–0.997 across all MV3 deployments vs 0.40–0.98 (erratic) for MV2 | ✅ Verified (logs); MV2 divergence undiagnosed |
| 5 | KD (→MV2) improves accuracy by 1–2 pp consistently in direction (4/4 runs) but below unpaired significance | ⚠️ Pending paired tests |
| 6 | The "INT8 above FP32" accuracy anomaly is unconfirmed — built on a cross-set comparison; like-for-like FP32@200 evaluation pending | ⚠️ Open question |
| 7 | Local (ONNX) NSE is necessary but not sufficient: models passing the local gate failed on hardware (0.397, 0.711, 0.719) — HW NSE must be reported for every deployment | ✅ Strengthened vs v1 |

---

## Open items before manuscript

1. **Re-extract MV2 baseline INT8 board accuracy** (records hold `null`; v1's 76.0% is untraceable).
2. **Evaluate all FP32 ONNX models on the 200-batch** → resolves or kills the Phase 5/6 "anomaly."
3. **Add McNemar paired tests + Wilson CIs to Final_Results** (predictions already collected in cell 7) → determines which accuracy claims survive.
4. **Diagnose MV2 INT8 hardware divergence** (HW NSE 0.40–0.72 on three deployments) before any MV2 board comparison is cited.
5. **Re-evaluate rebuilt checkpoints** (best-val weights) to replace provisional accuracies.
6. **Confirm test-set-fix timeline**: was the `test_input.npz` used in the 4–5 June board runs the post-fix version? If not, those board accuracies must be re-run.
7. **Resolve quantization granularity**: ORT exports per-channel weights; stedgeai reports "sa/sa per tensor" — determine what Neural ART compiled.
8. Preserve all `output_*.npz` board dumps → enables paired hardware statistics.

---

## Toolchain

PyTorch · ONNX Runtime (static QDQ, per-channel weights / per-tensor activations) · ST Edge AI Core 10.2.0 (`stedgeai`, Neural ART, mode=TARGET via serial) · Google Colab.
**Hardware:** STM32N6570-DK · Cortex-M55 @800 MHz · Neural ART NPU @1 GHz · OctoFlash 112 MB · HyperRAM 32 MB.

*Università degli Studi di Milano · IEBI Lab · June 2026 — v2*
