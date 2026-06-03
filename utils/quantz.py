# =====================================================
# utils/quantz.py
# Shared quantisation helpers — used by:
#   Model_KD_Combined, Model_Pruning_Combined, Pipeline_Quantz
#
# Functions:
#   generate_shared_test_files  — balanced 100+100 sample test set
#   export_onnx                 — PyTorch → FP32 ONNX
#   save_calib_npz              — calibration data from train loader
#   quantize_int8               — FP32 ONNX → INT8 QDQ ONNX
#   compute_nse                 — FP32 vs INT8 logit agreement
#   compute_local_int8_accuracy — accuracy via ONNX Runtime (Pipeline_Quantz)
#   compute_stm32_accuracy      — accuracy from STM32 CLI output  (Pipeline_Quantz)
# =====================================================

import numpy as np
from pathlib import Path

import torch

# onnx / onnxruntime installed at notebook runtime via:
#   !pip -q install onnx onnxruntime onnxruntime-tools
# Import lazily inside each function so the module loads even before pip runs.


# ── Shared test files ────────────────────────────────────────────────

def generate_shared_test_files(shared_dir, test_loader, n_per_class=100):
    """
    Build a class-balanced evaluation set: n_per_class non-person +
    n_per_class person samples.  Saved once; skipped on re-runs.

    IMPORTANT: test_loader must have shuffle=False and the dataset must
    have VWWDataset's ordering (non-person first).  The function collects
    samples until it has n_per_class from each class label, so ordering
    does not matter as long as both classes are reachable.

    Returns (inp_path, lbl_path).
    """
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    inp_p = shared_dir / "test_input.npz"
    lbl_p = shared_dir / "test_labels.npz"
    if inp_p.exists() and lbl_p.exists():
        print("    ⏭️  Shared test files exist")
        return inp_p, lbl_p
    per_class = {0: [], 1: []}
    for x, y in test_loader:
        lbl = int(y.item())
        if len(per_class[lbl]) < n_per_class:
            per_class[lbl].append(x.numpy().astype("float32")[0])
        if all(len(v) == n_per_class for v in per_class.values()):
            break
    inputs = per_class[0] + per_class[1]
    labels = [0] * n_per_class + [1] * n_per_class
    np.savez(inp_p, input=np.stack(inputs))
    np.savez(lbl_p, label=np.array(labels, dtype="int32"))
    print(f"    ✅ Shared test files saved ({n_per_class * 2} samples, balanced)")
    return inp_p, lbl_p


# ── ONNX export ──────────────────────────────────────────────────────

def export_onnx(model, path, device):
    """Export model to FP32 ONNX (opset 18). Skips if file already exists."""
    import onnx as _onnx
    if Path(path).exists():
        print("    ⏭️  FP32 ONNX exists"); return
    model.eval()
    dummy = torch.randn(1, 3, 96, 96, device=device)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        export_params=True, opset_version=18,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )
    _onnx.checker.check_model(str(path), full_check=False)
    print("    ✅ FP32 ONNX saved")


# ── Calibration data ─────────────────────────────────────────────────

def save_calib_npz(path, train_loader, n=200):
    """Save n random training samples to NPZ for INT8 calibration."""
    if Path(path).exists():
        print("    ⏭️  Calib data exists"); return
    xs = []
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= n: break
            xs.append(x.numpy().astype("float32")[0])
    np.savez(path, input=np.stack(xs))
    print(f"    ✅ Calib data saved ({n} samples)")


# ── INT8 quantisation ────────────────────────────────────────────────

class _CalibReader:
    """Internal calibration reader — not part of the public API."""
    def __init__(self, npz_path):
        from onnxruntime.quantization import CalibrationDataReader
        # dynamic inheritance so onnxruntime isn't imported at module load
        self.__class__ = type(
            "_CalibReader",
            (CalibrationDataReader,),
            dict(self.__class__.__dict__),
        )
        self.data = np.load(npz_path)["input"].astype("float32")
        self.i    = 0

    def get_next(self):
        if self.i >= len(self.data): return None
        out = {"input": self.data[self.i:self.i + 1]}
        self.i += 1
        return out

    def rewind(self):
        self.i = 0


def quantize_int8(fp32_path, calib_path, int8_path):
    """
    Static INT8 QDQ quantisation via ONNX Runtime.
    Skips if int8_path already exists.
    """
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationDataReader,
    )

    if Path(int8_path).exists():
        print("    ⏭️  INT8 ONNX exists"); return

    class _Reader(CalibrationDataReader):
        def __init__(self, path):
            self.data = np.load(path)["input"].astype("float32")
            self.i = 0
        def get_next(self):
            if self.i >= len(self.data): return None
            out = {"input": self.data[self.i:self.i + 1]}; self.i += 1; return out
        def rewind(self): self.i = 0

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=_Reader(calib_path),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print("    ✅ INT8 QDQ ONNX saved")


# ── NSE validation ───────────────────────────────────────────────────

def compute_nse(fp32_path, int8_path, input_npz):
    """
    Nash-Sutcliffe Efficiency between FP32 and INT8 ONNX logits.
    NSE ≥ 0.95: quantisation distortion is small — safe to deploy.
    NSE < 0.95: significant logit shift — exclude from Pipeline_Quantz.
    Returns float('nan') if FP32 logits are near-constant (degenerate model).
    """
    import onnxruntime as ort
    inputs = np.load(input_npz)["input"]
    s32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    s8  = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    fp32_outs, int8_outs = [], []
    for i in range(len(inputs)):
        sample = inputs[i:i + 1]
        fp32_outs.append(s32.run(["logits"], {"input": sample})[0][0])
        int8_outs.append(s8.run(["logits"],  {"input": sample})[0][0])
    fp32_outs = np.array(fp32_outs)
    int8_outs = np.array(int8_outs)
    num = np.sum((fp32_outs - int8_outs) ** 2)
    den = np.sum((fp32_outs - fp32_outs.mean()) ** 2)
    if den < 1e-8:
        return float("nan")   # degenerate: model outputs near-constant logits
    return float(1 - num / den)


# ── Accuracy helpers (Pipeline_Quantz only) ──────────────────────────

def compute_local_int8_accuracy(int8_path, input_npz, labels_npz):
    """
    Accuracy of INT8 ONNX model on the shared balanced test set.
    Returns percentage float.  Used in Pipeline_Quantz for quick local check.
    """
    import onnxruntime as ort
    sess   = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    inputs = np.load(input_npz)["input"]
    labels = np.load(labels_npz)["label"].astype("int64")
    preds  = [
        int(np.argmax(sess.run(["logits"], {"input": inputs[i:i + 1]})[0][0]))
        for i in range(len(inputs))
    ]
    return float((np.array(preds) == labels).mean() * 100)


def compute_stm32_accuracy(labels_npz, outputs_npz,
                           key="c_outputs_1", num_classes=2):
    """
    Accuracy from STM32 CLI output NPZ file.
    Loads raw logits, argmax, compares to shared test labels.
    Returns percentage float.
    """
    labels = np.load(labels_npz)["label"].astype("int64")
    raw    = np.load(outputs_npz)[key]
    if raw.size != len(labels) * num_classes:
        raise ValueError(f"Size mismatch: {raw.size} vs {len(labels) * num_classes}")
    preds = np.argmax(raw.reshape(len(labels), num_classes), axis=1)
    return float((preds == labels).mean() * 100)
