# Optimization of Deep Learning Models for Edge Deployment

**Master's Thesis — Università degli Studi di Milano**  
IEBI Lab, Department of Computer Science  
Supervisor: Angelo Genovese  
Target hardware: STM32N6570-DK (ST Neural ART NPU)  
Task: Binary person detection (Visual Wake Words)

---

## Overview

This thesis investigates model compression techniques — quantization, knowledge distillation, and pruning — for deploying lightweight convolutional neural networks on the STM32N6 microcontroller. The full pipeline covers PyTorch training → ONNX export → INT8 static quantization → on-device validation via `stedgeai`.

**Models:** Custom truncated MobileNetV2 (151,874 params) and MobileNetV3 (139,428 params)  
**Dataset:** Visual Wake Words (VWW) — binary person/non-person detection at 96×96  
**Toolchain:** PyTorch · ONNX Runtime · ST Edge AI Core (`stedgeai`) · Google Colab

---

## Repository structure

```
thesis-v1/
├── utils/
│   ├── dataset.py          # VWW dataset loader, augmentation, manifest handling
│   ├── models.py           # MobileNetV2, MobileNetV3, VGGStyle, ResNet, teachers
│   ├── train.py            # Training loop, early stopping, KD loss
│   └── quantz.py           # INT8 static quantization pipeline, NSE validation
├── vww_fixed_split_manifests/  # Fixed train/val/test file lists (seed 41)
├── outputs/                # JSON records for all experiments
├── logs/                   # STM32 generate + validate logs for all deployed models
├── Model_MobileNetV2.ipynb
├── Model_MobileNetV3.ipynb
├── Model_VGG.ipynb
├── Model_ResNet.ipynb
├── Model_VGG_Pretrained.ipynb
├── Model_ResNet_Pretrained.ipynb
├── Model_KD_Combined.ipynb
├── Model_Pruning_Combined.ipynb
├── Pipeline_Quantz.ipynb   # Universal INT8 quantization + NSE gate
└── Final_Results.ipynb     # Held-out test set evaluation
```

---

## Phase 1 — Dataset

**Source:** Silicon Labs VWW (COCO2014-based, 96×96 pre-resized)  
**Split:** Fixed 70/15/15, seed 41, saved as manifest files

| Split | Samples | Per class | Purpose |
|---|---|---|---|
| Train | 7,000 | 3,500 | Model training |
| Validation | 1,500 | 750 | Checkpoint selection, early stopping |
| Test (held-out) | 1,500 | 750 | Final evaluation only |
| STM32 batch | 200 | 100 | Shared hardware eval across all models |

All splits are perfectly balanced (50/50 person/non-person). The 200-sample STM32 batch is a fixed balanced subset drawn from the test set — identical across all models for fair comparison.

**Augmentation:** RandomResizedCrop(96, scale 0.7–1.0) · RandomHorizontalFlip · RandomRotation ±15° · ColorJitter(0.2) · ImageNet normalize

---

## Phase 2 — Baseline training

**Training config (scratch models):** Adam · LR 1e-3 · batch 64 · CosineAnnealingLR · 50 epochs · patience 10 · label smoothing 0.1  
**Teacher config (pretrained):** 3-phase unfreeze · 30 epochs · LR 3e-4 → 1e-4 → 3e-5

### Accuracy results

| Model | Role | Params | Val% (1,500) | Test% (1,500) | STM32 FP32% (200) | STM32 INT8% (200) | Local NSE |
|---|---|---|---|---|---|---|---|
| MobileNetV2 | student | 151,874 | 78.40 | 79.47 | 76.50 | 76.00 | 0.9777 |
| MobileNetV3 | student | 139,428 | 79.13 | 79.13 | 79.50 | 80.00 | 0.9931 |
| VGGStyle (scratch) | comparison only | 5,958,242 | 80.27 | — | — | — | — |
| ResNet (scratch) | comparison only | 2,828,194 | 77.93 | — | — | — | — |
| VGG16-BN (pretrained) | teacher | 27,634,626 | 89.07 | 90.60 | — | — | — |
| ResNet50 (pretrained) | teacher | 24,623,042 | 87.93 | 89.73 | — | — | — |

> VGGStyle and ResNet scratch: val acc = best-seed result (seed 41 and seed 63 respectively). Not evaluated on test set. Not deployed to hardware. Trained for accuracy comparison with pretrained teachers only.  
> Teacher param counts reflect modified classifier heads (original ImageNet output layer replaced).

### Hardware: FP32 vs INT8 on STM32N6

| Model | Format | Latency | Throughput | Flash | RAM | NPU epochs |
|---|---|---|---|---|---|---|
| MobileNetV2 | FP32 | 538.8 ms | 1.86 inf/s | 580.9 KiB | 906.4 KiB | 1 / 46 HW ❌ |
| MobileNetV2 | INT8 | **2.833 ms** | **352.9 inf/s** | **154.5 KiB** | **242.1 KiB** | 25 / 25 HW ✅ |
| MobileNetV3 | FP32 | 385.4 ms | 2.59 inf/s | 544.7 KiB | 679.1 KiB | 3 / 96 HW ❌ |
| MobileNetV3 | INT8 | **3.659 ms** | **273.3 inf/s** | **141.7 KiB** | **216.0 KiB** | 55 / 55 HW ✅ |

> **Key finding:** INT8 triggers full NPU offload on the STM32N6. MV2 gains a **190× latency reduction**; MV3 a **105× reduction**. Flash and RAM drop ~73%. FP32 runs almost entirely in software (slow); INT8 is the only viable format for real-time edge deployment on this hardware.

---

## Phase 3 — Knowledge distillation

**Config:** T=4.0 · α=0.7 · 80 epochs · patience 20  
**8 runs:** 2 teachers × 2 students × 2 init strategies (ft = warm-start, scratch = random init)

| Run | Teacher | Student | Init | Val% (1,500) | Test% (1,500) | Δ | NSE | STM32 INT8% (200) |
|---|---|---|---|---|---|---|---|---|
| vgg_mv2_ft | VGG16-BN | MV2 | ft | 80.07 | 81.00 | +1.67 | 0.987 | 80.0 ✅ |
| vgg_mv2_scratch | VGG16-BN | MV2 | scratch | 79.53 | 80.40 | +1.13 | 0.983 | 79.0 ✅ |
| vgg_mv3_ft | VGG16-BN | MV3 | ft | 79.13 | — | 0.00 | 0.991 | — |
| vgg_mv3_scratch | VGG16-BN | MV3 | scratch | 79.53 | — | +0.40 | 0.992 | — |
| resnet_mv2_ft | ResNet50 | MV2 | ft | 79.47 | 80.67 | +1.07 | 0.961 | 82.0 ⚠️ |
| resnet_mv2_scratch | ResNet50 | MV2 | scratch | 79.53 | 81.33 | +1.13 | 0.978 | 84.0 ⚠️ |
| resnet_mv3_ft | ResNet50 | MV3 | ft | 79.13 | — | 0.00 | 0.993 | — |
| resnet_mv3_scratch | ResNet50 | MV3 | scratch | 79.53 | — | +0.40 | 0.987 | — |

> ✅ **VGG-distilled MV2** models deploy reliably — INT8 tracks val accuracy closely. `vgg_mv2_ft` achieves 80.0% STM32 INT8, a +4 pp gain over the MV2 baseline at no latency cost.  
> ⚠️ **ResNet-distilled MV2** models show anomalously inflated INT8 accuracy (82–84% vs ~79.5% val). Attributed to miscalibrated per-tensor static quantization. Same pattern as structured pruning (Phase 5). Not selected as deployment candidates.  
> MV3 KD models were not deployed to hardware. MV3 showed no improvement with the VGG teacher (0.00% delta) and marginal improvement (+0.40%) from scratch — MV3 appears resistant to KD supervision.

---

## Phase 4 — Unstructured pruning

**Method:** L1 magnitude-based, all Conv2d layers · 3 sparsity levels · 10-epoch FT (LR 1e-4)  
**All 6 checkpoints passed the NSE gate (≥0.95) and were deployed to STM32 INT8.**

| Model | Target | Post-prune% (1,500) | After FT% (1,500) | Δ | NZ reduction | NSE | STM32 INT8% (200) | HW NSE |
|---|---|---|---|---|---|---|---|---|
| MV2 | 10% | 78.47 | 78.93 | +0.53 | 9.5% | 0.9816 | 78.5 | 0.957 |
| MV2 | 20% | 77.27 | 78.53 | +0.13 | 19.0% | 0.9844 | 79.5 | 0.956 |
| MV2 | 30% | 71.33 | 78.27 | −0.13 | 28.5% | 0.9800 | 76.0 | 0.954 |
| MV3 | 10% | 78.93 | 78.67 | −0.47 | 7.3% | 0.9840 | 80.5 | 0.995 |
| MV3 | 20% | 76.53 | 78.27 | −0.87 | 14.7% | 0.9871 | 81.0 | 0.995 |
| MV3 | 30% | 65.87 | 78.47 | −0.67 | 22.0% | 0.9892 | 83.0 | 0.995 |

**Hardware comparison — pruned vs baseline INT8:**

| Model | Latency | Flash | RAM | MACCs |
|---|---|---|---|---|
| MV2 Baseline INT8 | 2.833 ms | 154.5 KiB | 242.1 KiB | 25,663,842 |
| MV2 Unstr 10% | 2.833 ms | 154.5 KiB | 242.1 KiB | 25,663,842 |
| MV2 Unstr 20% | 2.833 ms | 154.5 KiB | 242.1 KiB | 25,663,842 |
| MV2 Unstr 30% | 2.833 ms | 154.5 KiB | 242.1 KiB | 25,663,842 |
| MV3 Baseline INT8 | 3.659 ms | 141.7 KiB | 216.0 KiB | 20,387,164 |
| MV3 Unstr 10% | 3.665 ms | 141.7 KiB | 216.0 KiB | 20,387,164 |
| MV3 Unstr 20% | 3.666 ms | 141.7 KiB | 216.0 KiB | 20,387,164 |
| MV3 Unstr 30% | 3.668 ms | 141.7 KiB | 216.0 KiB | 20,387,164 |

> ❌ **Confirmed negative finding:** Every hardware metric is identical across all sparsity levels and the unpruned baseline. The STM32N6 Neural ART NPU executes dense INT8 regardless of weight sparsity — zero weights are not compressed, skipped, or treated differently. Unstructured pruning achieves theoretical parameter reduction but provides **zero hardware efficiency benefit** on this deployment target.

---

## Phase 5 — Structured pruning

**Method:** L2 filter-level, pointwise Conv2d only (depthwise skipped for channel alignment) · 3 levels · 15-epoch FT

### NSE gate results

| Model | Target | Actual | Post-prune% (1,500) | After FT% (1,500) | Δ | NSE | Gate | Deployed |
|---|---|---|---|---|---|---|---|---|
| MV2 | 2% | 1.8% | 75.20 | 79.33 | +0.93 | 0.9324 | ❌ FAIL | no |
| MV2 | 3% | 2.9% | 75.67 | 78.73 | +0.33 | 0.9257 | ❌ FAIL | no |
| MV2 | 5% | 4.7% | 70.27 | 78.53 | +0.13 | 0.8471 | ❌ FAIL | no |
| MV3 | 2% | 1.8% | 76.20 | 78.87 | −0.27 | 0.9860 | ✅ pass | yes |
| MV3 | 3% | 2.9% | 71.33 | 78.73 | −0.40 | 0.9728 | ✅ pass | yes |
| MV3 | 5% | 4.7% | 66.93 | 78.33 | −0.80 | 0.8252 | ❌ FAIL | no |

### Deployed models — STM32 results

| Model | FP32 val% (1,500) | Local INT8% (200) | STM32 INT8% (200) | vs baseline | Latency | Flash | HW NSE |
|---|---|---|---|---|---|---|---|
| MV3 Structured 2% | 78.87 | 83.5 | 83.0 | +3.87 pp ⚠️ | 3.668 ms | 141.7 KiB | 0.997 |
| MV3 Structured 3% | 78.73 | 83.5 | 84.5 | +5.77 pp ⚠️ | 3.661 ms | 141.7 KiB | 0.996 |
| MV3 Baseline INT8 (ref) | 79.13 | 81.0 | 80.0 | +0.87 pp | 3.659 ms | 141.7 KiB | 0.997 |

> ❌ **Confirmed negative finding:** INT8 accuracy for both deployed structured models is 4–6 pp **above** their own FP32 val accuracy — physically anomalous. A well-calibrated INT8 model should perform equal to or slightly below FP32. Root cause: L2 filter removal permanently changes the activation distributions of all downstream layers; per-tensor static calibration cannot characterise these shifted statistics, producing a miscalibrated decision boundary that inflates apparent accuracy on the small 200-sample batch.  
> High HW NSE (0.996–0.997) confirms the model deploys faithfully — the problem is the model itself, not the hardware execution. No hardware efficiency benefit exists either: latency, memory, and MACCs are unchanged at these sparsity levels (1.8–2.9%).  
> **Structured pruning + per-tensor static INT8 quantization is incompatible for this pipeline.**

---

## Complete results — all deployed models

| Model | Technique | Val% (1,500) | Test% (1,500) | STM32 FP32% (200) | STM32 INT8% (200) | Local NSE |
|---|---|---|---|---|---|---|
| MobileNetV2 | baseline | 78.40 | 79.47 | 76.50 | 76.00 | 0.978 |
| MobileNetV3 | baseline | 79.13 | 79.13 | 79.50 | 80.00 | 0.993 |
| MV2 KD VGG-ft ✅ | KD | 80.07 | 81.00 | — | 80.00 | 0.987 |
| MV2 KD VGG-scratch ✅ | KD | 79.53 | 80.40 | — | 79.00 | 0.983 |
| MV2 KD ResNet-ft ⚠️ | KD | 79.47 | 80.67 | — | 82.00 | 0.961 |
| MV2 KD ResNet-scratch ⚠️ | KD | 79.53 | 81.33 | — | 84.00 | 0.978 |
| MV2 Unstr 10% | pruning | 78.93 | 80.40 | — | 78.50 | 0.982 |
| MV2 Unstr 20% | pruning | 78.53 | 79.73 | — | 79.50 | 0.984 |
| MV2 Unstr 30% | pruning | 78.27 | 78.47 | — | 76.00 | 0.980 |
| MV3 Unstr 10% | pruning | 78.67 | 79.13 | — | 80.50 | 0.984 |
| MV3 Unstr 20% | pruning | 78.27 | 79.07 | — | 81.00 | 0.987 |
| MV3 Unstr 30% | pruning | 78.47 | 79.07 | — | 83.00 | 0.989 |
| MV3 Struct 2% ⚠️ | pruning | 78.87 | 79.47 | — | 83.00 | 0.986 |
| MV3 Struct 3% ⚠️ | pruning | 78.73 | 79.67 | — | 84.50 | 0.973 |

> ⚠️ = INT8 accuracy anomalously above FP32 val accuracy by 4–6 pp — miscalibrated per-tensor static quantization artifact. Not reliable indicators of generalization performance.

---

## Key findings

| # | Finding | Result |
|---|---|---|
| 1 | **INT8 quantization** is the single most impactful optimization — triggers full NPU offload, 105–190× speedup, ~73% memory reduction | ✅ Positive |
| 2 | **KD (VGG teacher → MV2)** achieves best INT8 accuracy: 80.0% vs 76.0% baseline, zero latency/memory cost | ✅ Positive |
| 3 | **MV3 > MV2** as a deployment architecture: better baseline accuracy, near-perfect HW NSE (0.997 vs 0.711), more stable quantization | ✅ Positive |
| 4 | **Unstructured pruning** has zero hardware benefit on the STM32N6 NPU — dense INT8 execution ignores weight sparsity | ❌ Negative |
| 5 | **Structured pruning + static INT8** is fundamentally incompatible — filter removal shifts activation distributions that per-tensor calibration cannot recover | ❌ Negative |
| 6 | **NSE is necessary but not sufficient** for hardware reliability — must be evaluated alongside classification accuracy | ⚠️ Nuance |

---

## Tools & dependencies

| Tool | Purpose |
|---|---|
| PyTorch | Model training, KD, pruning |
| ONNX Runtime | FP32 → INT8 static quantization (QDQ format) |
| ST Edge AI Core (`stedgeai`) | STM32 model generation, validation, profiling |
| Google Colab | All training and quantization experiments |
| Google Drive | Checkpoint and manifest persistence |

**Hardware:** STM32N6570-DK · ST Neural ART NPU · 800 MHz Cortex-M55 · 1000 MHz NPU · OctoFlash 112 MB · HyperRAM 32 MB

---

*Università degli Studi di Milano · IEBI Lab · June 2026*
