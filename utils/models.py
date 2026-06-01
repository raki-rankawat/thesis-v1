# =====================================================
# utils/models.py
# All model definitions for the STM32 thesis
#
# Models:
#   VWW_MobileNetV2  — truncated MobileNetV2, primary deployment model
#   VWW_MobileNetV3  — truncated MobileNetV3-Small style, SE + h-swish
#   VWW_VGGStyle     — custom 4-block VGG, used as teacher in KD
#   VWW_ResNet       — custom lightweight ResNet, comparison baseline
#
# NOTE: VWW_MobileNetV2 and VWW_MobileNetV3 are custom truncated variants
# designed for 96×96 VWW input and STM32 memory constraints.
# Do not compare accuracy numbers to published MobileNetV3 benchmarks.
# =====================================================

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16_bn, VGG16_BN_Weights


# ── Shared weight init ────────────────────────────────

def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


# ── MobileNetV2 ───────────────────────────────────────

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return x + self.block(x) if self.use_residual else self.block(x)


class VWW_MobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
        )
        self.features = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6), InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 2, 6), InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 2, 6), InvertedResidual(64, 64, 1, 6),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 512, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(512, num_classes)
        _init_weights(self)

    def forward(self, x):
        x = self.initial(x)
        x = self.features(x)
        x = self.head(x)
        return self.classifier(x.view(x.size(0), -1))


# ── MobileNetV3 ───────────────────────────────────────
#
# Key differences vs MobileNetV2:
#   • Hard-swish activation instead of ReLU6
#   • Hard-sigmoid inside SE blocks instead of sigmoid
#   • Squeeze-and-Excitation (SE) on selected layers
#   • Narrower expansion ratios → fewer params
#
# Architecture: 6 bottleneck blocks, 256-wide final conv
# Input: 96×96×3   Output: 2-class logits

class _HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class _HardSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation: recalibrates channel responses."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeezed, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, channels, 1, bias=True),
            _HardSigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class _V3Bottleneck(nn.Module):
    """
    MobileNetV3 bottleneck.
    in_ch, exp_ch, out_ch : channel sizes
    stride                : 1 (same) or 2 (downsample)
    use_se                : add Squeeze-and-Excitation
    use_hs                : hard-swish (True) or ReLU (False)
    """
    def __init__(self, in_ch, exp_ch, out_ch, stride, use_se, use_hs):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        act = _HardSwish() if use_hs else nn.ReLU(inplace=True)
        layers = []
        if exp_ch != in_ch:
            layers += [nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                       nn.BatchNorm2d(exp_ch), act]
        layers += [
            nn.Conv2d(exp_ch, exp_ch, 3, stride=stride, padding=1,
                      groups=exp_ch, bias=False),
            nn.BatchNorm2d(exp_ch), act,
        ]
        if use_se:
            layers.append(_SEBlock(exp_ch))
        layers += [
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_residual else self.block(x)


class VWW_MobileNetV3(nn.Module):
    """
    Truncated MobileNetV3-Small style for 96x96 VWW.
    Comparable parameter count to VWW_MobileNetV2 for fair comparison.

    Block config: (in_ch, exp_ch, out_ch, stride, use_se, use_hs)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            _HardSwish(),
        )
        cfg = [
            (16,  16,  16, 1, False, False),
            (16,  72,  24, 2, False, False),
            (24,  88,  24, 1, False, False),
            (24,  96,  40, 2, True,  True),
            (40, 240,  40, 1, True,  True),
            (40, 120,  48, 1, True,  True),
        ]
        self.features = nn.Sequential(*[_V3Bottleneck(*c) for c in cfg])
        self.head = nn.Sequential(
            nn.Conv2d(48, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            _HardSwish(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            _HardSwish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        _init_weights(self)

    def forward(self, x):
        x = self.initial(x)
        x = self.features(x)
        x = self.head(x)
        return self.classifier(x.view(x.size(0), -1))


# ── VGG-Style ─────────────────────────────────────────

class VWW_VGGStyle(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 96→48
            nn.Conv2d(3,   32,  3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  32,  3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 48→24
            nn.Conv2d(32,  64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 24→12
            nn.Conv2d(64,  128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 12→6
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 128),          nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        _init_weights(self)

    def forward(self, x):
        return self.classifier(self.features(x))
    

# ── VGG-Style Pretrained ────────────────────────────────────────────

class VGG_Pretrained(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512,   128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        for p in self.features.parameters():
            p.requires_grad = False   # backbone frozen initially

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0), -1))

    def unfreeze_top(self):
        for p in self.features[24:].parameters():
            p.requires_grad = True
        print("🔥 VGG16-BN: unfroze features[24:]")

    def unfreeze_all(self):
        for p in self.features.parameters():
            p.requires_grad = True
        print("🔥 VGG16-BN: unfroze all features")



# ── ResNet50 Pretrained ──────────────────────────────────────────────

class ResNet_Pretrained(nn.Module):
    """
    ResNet50 fine-tuned from ImageNet weights on VWW.

    Progressive unfreeze strategy (mirrors VGG_Pretrained):
      Phase 1: backbone fully frozen  → train head only
      Phase 2: unfreeze layer3+layer4 → mid-level features adapt
      Phase 3: unfreeze all           → full fine-tune at low LR

    Head: 2048 → 512 → 128 → 2  (replaces ImageNet fc)
    Input: 96×96×3 (works; ResNet downsamples aggressively enough)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Drop the original fc; keep everything else as backbone
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512,  128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        # Phase 1: freeze entire backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x.view(x.size(0), -1))

    def unfreeze_top(self):
        """Phase 2: unfreeze layer3 and layer4 (indices 6 and 7 in backbone)."""
        for p in self.backbone[6].parameters():   # layer3
            p.requires_grad = True
        for p in self.backbone[7].parameters():   # layer4
            p.requires_grad = True
        print("🔥 ResNet50: unfroze layer3 + layer4")

    def unfreeze_all(self):
        """Phase 3: unfreeze entire backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("🔥 ResNet50: unfroze all backbone layers")


# ── ResNet ────────────────────────────────────────────

class _BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential() if (stride == 1 and in_ch == out_ch) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class VWW_ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self._in  = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = self._make(32,  2, stride=1)
        self.layer2 = self._make(64,  2, stride=2)
        self.layer3 = self._make(128, 2, stride=2)
        self.layer4 = self._make(256, 2, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        _init_weights(self)

    def _make(self, out_ch, n, stride):
        layers = [_BasicBlock(self._in, out_ch, stride)]
        self._in = out_ch
        for _ in range(1, n):
            layers.append(_BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.classifier(self.pool(x))


# ── Helpers ───────────────────────────────────────────

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model, path="/tmp/_size_check.pth"):
    import os, torch
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1e6
    os.remove(path)
    return size