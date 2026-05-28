# =====================================================
# utils/train.py
# Standardized training loops, KD loss, multi-seed runner
#
# STANDARD HYPERPARAMETERS (used everywhere):
#   batch_size      : 64
#   optimizer       : Adam
#   weight_decay    : 1e-4
#   label_smoothing : 0.1
#   scheduler       : CosineAnnealingLR
#   patience        : 10
#   augmentation    : "standard"
#   seeds           : [41, 52, 63]
#
#   Scratch models  : lr=1e-3, epochs=50
#   Pretrained models: lr=3e-4, epochs=30 (converge faster)
# =====================================================

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Reproducibility ───────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_device(seed=41, deterministic=True):
    set_seed(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


# ── Epoch-level functions ─────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out  = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs        = y.size(0)
        loss_sum += loss.item() * bs
        correct  += (out.argmax(1) == y).sum().item()
        total    += bs
    return loss_sum / total, correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out  = model(X)
            loss = criterion(out, y)
            bs        = y.size(0)
            loss_sum += loss.item() * bs
            correct  += (out.argmax(1) == y).sum().item()
            total    += bs
    return loss_sum / total, correct / total


def evaluate(model, loader, device):
    """Returns accuracy as float in [0, 1]."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (model(X).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total


# ── Standard training loop (scratch models) ───────────

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    *,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=10,
    save_path,
    verbose=True,
):
    """
    Standard training loop for scratch models.
    Single optimizer, CosineAnnealingLR, early stopping.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []
    best_acc, best_epoch, patience_ctr = 0.0, 0, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tl); train_accs.append(ta)
        val_losses.append(vl);   val_accs.append(va)

        if verbose:
            marker = " ✅" if va > best_acc else ""
            print(f"Epoch {epoch:3d}/{epochs} | LR {scheduler.get_last_lr()[0]:.6f} | "
                  f"Train {ta*100:.2f}% | Val {va*100:.2f}%{marker}")

        if va > best_acc:
            best_acc, best_epoch, patience_ctr = va, epoch, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"🛑 Early stopping at epoch {epoch}  (best: {best_acc*100:.2f}% @ ep {best_epoch})")
            break

    elapsed = (time.time() - start) / 60
    print(f"\n✅ Best val acc: {best_acc*100:.2f}% @ epoch {best_epoch}  ({elapsed:.1f} min)")
    print(f"   Saved: {save_path}")
    return {
        "best_acc": best_acc, "best_epoch": best_epoch,
        "train_losses": train_losses, "train_accs": train_accs,
        "val_losses":   val_losses,   "val_accs":   val_accs,
        "elapsed_min":  elapsed,      "save_path":  str(save_path),
    }


# ── Three-phase training loop (pretrained models) ─────

def train_model_three_phase(
    model,
    train_loader,
    val_loader,
    device,
    *,
    epochs=30,
    lr_phase1=3e-4,
    lr_phase2=1e-4,
    lr_phase3=3e-5,
    phase2_epoch=10,
    phase3_epoch=20,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=10,
    save_path,
    verbose=True,
):
    """
    Three-phase progressive unfreeze training for pretrained models.

    Phase 1 (epochs 1  → phase2_epoch-1): backbone frozen, train head only at lr_phase1
    Phase 2 (epochs phase2_epoch → phase3_epoch-1): partial unfreeze at lr_phase2
    Phase 3 (epochs phase3_epoch → end): full unfreeze at lr_phase3

    Model must implement unfreeze_top() and unfreeze_all() methods.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _make_optimizer(lr):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )

    optimizer = _make_optimizer(lr_phase1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epoch - 1)

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []
    best_acc, best_epoch, patience_ctr = 0.0, 0, 0
    current_phase = 1
    start = time.time()

    for epoch in range(1, epochs + 1):

        # Phase transitions
        if epoch == phase2_epoch and current_phase == 1:
            print(f"\n── Phase 2: partial unfreeze (epoch {epoch}) ──")
            if hasattr(model, "unfreeze_top"):
                model.unfreeze_top()
            elif hasattr(model, "unfreeze_layer4"):
                model.unfreeze_layer4()
            optimizer = _make_optimizer(lr_phase2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=phase3_epoch - phase2_epoch
            )
            current_phase = 2

        if epoch == phase3_epoch and current_phase == 2:
            print(f"\n── Phase 3: full unfreeze (epoch {epoch}) ──")
            if hasattr(model, "unfreeze_all"):
                model.unfreeze_all()
            optimizer = _make_optimizer(lr_phase3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - phase3_epoch + 1
            )
            current_phase = 3

        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tl); train_accs.append(ta)
        val_losses.append(vl);   val_accs.append(va)

        if verbose:
            marker = " ✅" if va > best_acc else ""
            print(f"[P{current_phase}] Epoch {epoch:3d}/{epochs} | LR {scheduler.get_last_lr()[0]:.6f} | "
                  f"Train {ta*100:.2f}% | Val {va*100:.2f}%{marker}")

        if va > best_acc:
            best_acc, best_epoch, patience_ctr = va, epoch, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"🛑 Early stopping at epoch {epoch}  (best: {best_acc*100:.2f}% @ ep {best_epoch})")
            break

    elapsed = (time.time() - start) / 60
    print(f"\n✅ Best val acc: {best_acc*100:.2f}% @ epoch {best_epoch}  ({elapsed:.1f} min)")
    print(f"   Saved: {save_path}")
    return {
        "best_acc": best_acc, "best_epoch": best_epoch,
        "train_losses": train_losses, "train_accs": train_accs,
        "val_losses":   val_losses,   "val_accs":   val_accs,
        "elapsed_min":  elapsed,      "save_path":  str(save_path),
    }


# ── Multi-seed runner ─────────────────────────────────

def train_multi_seed(
    model_fn,
    train_loader,
    val_loader,
    device,
    seeds=(41, 52, 63),
    save_dir=None,
    name_prefix="model",
    pretrained=False,
    **train_kwargs,
):
    """
    Trains model_fn() across multiple seeds.
    Uses train_model() for scratch, train_model_three_phase() for pretrained.
    Returns (results_list, best_result).
    """
    from pathlib import Path
    save_dir  = Path(save_dir or "/content/drive/My Drive/Colab Notebooks")
    train_fn  = train_model_three_phase if pretrained else train_model
    results   = []

    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        set_seed(seed)
        model     = model_fn().to(device)
        save_path = save_dir / f"{name_prefix}_seed_{seed}.pth"
        result    = train_fn(model, train_loader, val_loader, device,
                             save_path=save_path, **train_kwargs)
        result["seed"] = seed
        results.append(result)

    accs = [r["best_acc"] for r in results]
    best = max(results, key=lambda r: r["best_acc"])

    print(f"\n{'='*60}")
    print(f"Multi-seed summary  [{name_prefix}]")
    print(f"{'='*60}")
    for r in results:
        print(f"  Seed {r['seed']}: {r['best_acc']*100:.2f}% @ ep {r['best_epoch']}")
    print(f"  Mean ± Std: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  Best: seed {best['seed']}  {best['best_acc']*100:.2f}%")
    print(f"  Checkpoint: {best['save_path']}")
    return results, best


# ── Knowledge Distillation ────────────────────────────

class KDLoss(nn.Module):
    """
    Hinton (2015) knowledge distillation loss.
    total = alpha * soft_loss(T) + (1 - alpha) * hard_loss

    T (temperature): controls softness of teacher distribution.
      Higher T → softer targets → more inter-class information transferred.
      Recommended range: 2–8. We use T=4 (Hinton et al., 2015).

    alpha: weight on soft loss.
      Higher alpha → student learns more from teacher than ground truth.
      We ablate alpha ∈ {0.5, 0.7, 0.9} in notebook 09_KD_Ablation.
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss()
        self.kl    = nn.KLDivLoss(reduction="batchmean")

    def forward(self, s_logits, t_logits, labels):
        hard  = self.ce(s_logits, labels)
        soft  = self.kl(
            F.log_softmax(s_logits / self.T, dim=1),
            F.softmax(t_logits    / self.T, dim=1),
        ) * (self.T ** 2)
        total = self.alpha * soft + (1.0 - self.alpha) * hard
        return total, hard, soft


def train_kd(
    student,
    teacher,
    train_loader,
    val_loader,
    device,
    *,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    temperature=4.0,
    alpha=0.7,
    patience=10,
    save_path,
    verbose=True,
):
    """
    Knowledge distillation training loop.
    Teacher is fully frozen. Student trained with KDLoss.
    Returns (best_val_acc, elapsed_minutes).
    """
    # Freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    kd_loss   = KDLoss(temperature, alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, patience_ctr = evaluate(student, val_loader, device), 0
    torch.save(student.state_dict(), save_path)
    print(f"Initial student accuracy: {best_acc*100:.2f}%")

    start = time.time()

    for epoch in range(1, epochs + 1):
        student.train()
        loss_sum, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.no_grad():
                t_logits = teacher(X)
            s_logits = student(X)
            loss, _, _ = kd_loss(s_logits, t_logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct  += (s_logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        scheduler.step()
        val_acc   = evaluate(student, val_loader, device)
        train_acc = correct / total

        if verbose:
            marker = " ✅" if val_acc > best_acc else ""
            print(f"Epoch {epoch:3d}/{epochs} | Train {train_acc*100:.2f}% | Val {val_acc*100:.2f}%{marker}")

        if val_acc > best_acc:
            best_acc, patience_ctr = val_acc, 0
            torch.save(student.state_dict(), save_path)
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

    elapsed = (time.time() - start) / 60
    print(f"\n✅ Best KD val acc: {best_acc*100:.2f}%  ({elapsed:.1f} min)")
    return best_acc, elapsed


# ── Plotting ──────────────────────────────────────────

def plot_history(result, title=""):
    """Plot loss and accuracy curves from a train_model result dict."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(result["train_losses"], label="Train"); ax1.plot(result["val_losses"], label="Val")
    ax1.set_title(f"Loss  {title}"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
    ax2.plot([a*100 for a in result["train_accs"]], label="Train")
    ax2.plot([a*100 for a in result["val_accs"]],   label="Val")
    ax2.set_title(f"Accuracy  {title}"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("%")
    ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()
    print(f"Best: {result['best_acc']*100:.2f}% @ epoch {result['best_epoch']}  "
          f"({result['elapsed_min']:.1f} min)")
