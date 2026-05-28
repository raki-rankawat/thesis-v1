# =====================================================
# utils/dataset.py
# VWW dataset utilities — train / val / test splits
# =====================================================

import random
import tarfile
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Constants ─────────────────────────────────────────
VWW_URL      = "https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz"
BASE_DIR     = Path("/content/vww_work")
ARCHIVE_PATH = BASE_DIR / "vw_coco2014_96.tar.gz"
EXTRACT_DIR  = BASE_DIR / "extracted"
MANIFEST_DIR = Path("/content/drive/My Drive/vww_fixed_split_manifests")

N_PER_CLASS  = 5000
TRAIN_RATIO  = 0.70   # 3,500 per class → 7,000 total
VAL_RATIO    = 0.15   # 750   per class → 1,500 total
TEST_RATIO   = 0.15   # 750   per class → 1,500 total  ← held-out, touch only in notebook 13

IMG_SIZE       = 96
SEED           = 41
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]


# ── Download / Extract ────────────────────────────────

def download_vww():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 0:
        print("✅ VWW archive already downloaded"); return
    print("⬇️  Downloading VWW archive...")
    urlretrieve(VWW_URL, ARCHIVE_PATH)
    print("✅ Download complete:", ARCHIVE_PATH)


def _safe_extract(tar, path):
    path = Path(path).resolve()
    for m in tar.getmembers():
        if not str((path / m.name).resolve()).startswith(str(path)):
            raise RuntimeError("❌ Unsafe path in tar archive")
    tar.extractall(path)


def extract_vww():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if any(EXTRACT_DIR.iterdir()):
        print("✅ VWW already extracted"); return
    print("📦 Extracting VWW archive...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        _safe_extract(tar, EXTRACT_DIR)
    print("✅ Extraction complete:", EXTRACT_DIR)


def find_vww_root():
    for p in EXTRACT_DIR.rglob("person"):
        if p.is_dir() and (p.parent / "non_person").is_dir():
            return p.parent
    raise RuntimeError("❌ Could not find VWW dataset root")


# ── Manifests ─────────────────────────────────────────

def manifest_paths():
    return {
        "train_person":     MANIFEST_DIR / "train_person.txt",
        "val_person":       MANIFEST_DIR / "val_person.txt",
        "test_person":      MANIFEST_DIR / "test_person.txt",
        "train_non_person": MANIFEST_DIR / "train_non_person.txt",
        "val_non_person":   MANIFEST_DIR / "val_non_person.txt",
        "test_non_person":  MANIFEST_DIR / "test_non_person.txt",
    }


def manifests_ready():
    return all(p.exists() for p in manifest_paths().values())


def _list_images(folder):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts],
        key=str
    )


def _save_manifest(file_list, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(f) for f in file_list))


def load_manifest(path):
    return [Path(l.strip()) for l in path.read_text().splitlines() if l.strip()]


def create_fixed_split_manifests(src_root):
    if manifests_ready():
        print("✅ Manifests already exist:", MANIFEST_DIR); return

    print("🧩 Creating fixed train/val/test split manifests...")
    person_imgs    = _list_images(src_root / "person")
    nonperson_imgs = _list_images(src_root / "non_person")
    assert len(person_imgs) >= N_PER_CLASS and len(nonperson_imgs) >= N_PER_CLASS

    rng = random.Random(SEED)
    rng.shuffle(person_imgs)
    rng.shuffle(nonperson_imgs)

    person_sel    = person_imgs[:N_PER_CLASS]
    nonperson_sel = nonperson_imgs[:N_PER_CLASS]

    def split(lst):
        n_val  = int(len(lst) * VAL_RATIO)
        n_test = int(len(lst) * TEST_RATIO)
        test   = lst[:n_test]
        val    = lst[n_test:n_test + n_val]
        train  = lst[n_test + n_val:]
        return train, val, test

    p_train, p_val, p_test = split(person_sel)
    n_train, n_val, n_test = split(nonperson_sel)

    mp = manifest_paths()
    _save_manifest(p_train, mp["train_person"]);    _save_manifest(n_train, mp["train_non_person"])
    _save_manifest(p_val,   mp["val_person"]);      _save_manifest(n_val,   mp["val_non_person"])
    _save_manifest(p_test,  mp["test_person"]);     _save_manifest(n_test,  mp["test_non_person"])

    print(f"✅ Train: {len(p_train)+len(n_train)} | Val: {len(p_val)+len(n_val)} | Test: {len(p_test)+len(n_test)}")
    print("   Manifests saved at:", MANIFEST_DIR)


# ── Dataset ───────────────────────────────────────────

class VWWDataset(Dataset):
    classes      = ["non_person", "person"]
    class_to_idx = {"non_person": 0, "person": 1}

    def __init__(self, person_manifest, nonperson_manifest, transform=None):
        self.transform = transform
        self.samples = (
            [(p, 0) for p in load_manifest(nonperson_manifest)] +
            [(p, 1) for p in load_manifest(person_manifest)]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Transforms ────────────────────────────────────────

def get_train_transform(augmentation="standard"):
    """
    standard : moderate augmentation — used for all models (scratch + pretrained)
    strong   : heavier augmentation — optional for students on small data
    minimal  : flip + crop only     — used during KD training
    """
    if augmentation == "strong":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif augmentation == "minimal":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMG_SIZE, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # standard — used everywhere by default
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Loaders ───────────────────────────────────────────

def get_loaders(batch_size=64, augmentation="standard", num_workers=2):
    """Returns (train_loader, val_loader). Test loader is separate — use get_test_loader()."""
    mp = manifest_paths()
    train_ds = VWWDataset(mp["train_person"], mp["train_non_person"], get_train_transform(augmentation))
    val_ds   = VWWDataset(mp["val_person"],   mp["val_non_person"],   get_eval_transform())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {batch_size}")
    return train_loader, val_loader


def get_test_loader(batch_size=64, num_workers=2):
    """
    Returns the held-out test loader.
    ⚠️  Only call this in notebook 13_Final_Results.ipynb.
    Never use this during model selection or hyperparameter tuning.
    """
    mp = manifest_paths()
    test_ds = VWWDataset(mp["test_person"], mp["test_non_person"], get_eval_transform())
    loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Test: {len(test_ds)} samples  ⚠️  Use only for final evaluation")
    return loader


def prepare_dataset():
    """Full setup: download → extract → manifests. Call once at the top of any notebook."""
    print("1/4 Download");  download_vww()
    print("2/4 Extract");   extract_vww()
    print("3/4 Find root"); root = find_vww_root(); print("   Root:", root)
    print("4/4 Manifests"); create_fixed_split_manifests(root)
    return root
