"""
DenseNet-121 comparison models (guide's Review-1 request: implement the same
framework with a second backbone to judge how effective the EfficientNet-B0
results are).

This file is ADDITIVE ONLY. It does not modify src/model.py or anything else
in the existing project; it just reuses the same design so the comparison
isolates the BACKBONE:

  - same multi-scale idea: pool feature maps from three depths, concatenate
  - same shared 256-unit FC layer (ReLU + Dropout 0.3)
  - same hierarchical heads (Stage 1: Normal/Abnormal, Stage 2: Pneumonia/TB/COVID-19)
  - same freeze / "unfreeze last 3 children" fine-tuning logic

Tap points were chosen to match the SPATIAL scales of the EfficientNet taps
(features[2] / [5] / [8] -> 56x56 / 14x14 / 7x7 feature maps at 224x224 input):

    early : denseblock1  (256 ch, 56x56)
    mid   : denseblock3  (1024 ch, 14x14)
    deep  : final features after norm5 + ReLU (1024 ch, 7x7)

    concatenated multi-scale vector = 256 + 1024 + 1024 = 2304 dimensions

Two variants are provided:
    MultiScaleDenseNet      hierarchical (2 heads)  -> the "same framework, different backbone" model
    FlatMultiScaleDenseNet  single 4-way head       -> optional, mirrors FlatMultiScaleEfficientNet

Quick self-test (no data or checkpoint needed):
    python src/model_densenet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class _DenseNetMultiScaleBase(nn.Module):
    """Shared backbone + multi-scale feature extraction for both variants."""

    # DenseNet `features` children whose outputs are tapped via forward hooks.
    # The "deep" scale is the final feature map (after norm5), taken from the
    # backbone's return value, so it is not hooked.
    TAPS = {"denseblock1": "early", "denseblock3": "mid"}

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights).features

        self._features = {}
        for module_name, key in self.TAPS.items():
            getattr(self.backbone, module_name).register_forward_hook(self._make_hook(key))

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Read the concatenated dimension from a dummy pass instead of
        # hardcoding it (same approach as model.py).
        with torch.no_grad():
            self.eval()
            self.concat_dim = self._extract(torch.zeros(1, 3, 224, 224)).shape[1]
        self.train()

    def _make_hook(self, key):
        def hook(module, inp, out):
            self._features[key] = out
        return hook

    def _extract(self, x):
        """Image batch -> concatenated multi-scale feature vector [N, 2304]."""
        self._features = {}
        # torchvision's DenseNet applies ReLU to the last feature map before
        # pooling; do the same for the deep scale.
        deep = F.relu(self.backbone(x))  # also populates self._features via hooks
        pooled = [
            self.gap(self._features["early"]).flatten(1),
            self.gap(self._features["mid"]).flatten(1),
            self.gap(deep).flatten(1),
        ]
        return torch.cat(pooled, dim=1)

    def set_backbone_trainable(self, trainable: bool, unfreeze_last_n_blocks: int = None):
        """Same logic as MultiScaleEfficientNet.set_backbone_trainable().
        For DenseNet-121 the last 3 children of `features` are
        transition3, denseblock4 and norm5 (the deepest, most task-specific
        part), the analogue of EfficientNet's last 3 children."""
        for p in self.backbone.parameters():
            p.requires_grad = trainable

        if trainable and unfreeze_last_n_blocks:
            for p in self.backbone.parameters():
                p.requires_grad = False
            children = list(self.backbone.children())
            for block in children[-unfreeze_last_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.backbone.parameters())
        print(f"[{self.__class__.__name__}] Backbone trainable params: {n_trainable:,} / {n_total:,}")


def _shared_block(concat_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(concat_dim, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
    )


class MultiScaleDenseNet(_DenseNetMultiScaleBase):
    """Hierarchical model: identical heads to MultiScaleEfficientNet.
    forward() returns (stage1_logits [N,2], stage2_logits [N,3])."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__(pretrained=pretrained)
        self.shared = _shared_block(self.concat_dim)
        self.stage1_head = nn.Linear(256, 2)  # Normal vs Abnormal
        self.stage2_head = nn.Linear(256, 3)  # Pneumonia / TB / COVID-19

        print(f"[MultiScaleDenseNet] Concatenated multi-scale feature dim: {self.concat_dim}")
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def forward(self, x):
        shared = self.shared(self._extract(x))
        return self.stage1_head(shared), self.stage2_head(shared)


class FlatMultiScaleDenseNet(_DenseNetMultiScaleBase):
    """Flat variant: same backbone / taps / shared layer, one 4-way head.
    forward() returns logits [N,4] in FLAT_LABELS order."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True, num_classes: int = 4):
        super().__init__(pretrained=pretrained)
        self.shared = _shared_block(self.concat_dim)
        self.classifier = nn.Linear(256, num_classes)

        print(f"[FlatMultiScaleDenseNet] Concatenated multi-scale feature dim: {self.concat_dim}")
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def forward(self, x):
        return self.classifier(self.shared(self._extract(x)))


if __name__ == "__main__":
    # Self-test: python src/model_densenet.py   (no weights download needed)
    x = torch.randn(2, 3, 224, 224)

    m = MultiScaleDenseNet(pretrained=False, freeze_backbone=True)
    s1, s2 = m(x)
    print("hierarchical:", tuple(s1.shape), tuple(s2.shape))
    assert m.concat_dim == 2304, f"expected 2304, got {m.concat_dim}"
    assert s1.shape == (2, 2) and s2.shape == (2, 3)

    m.set_backbone_trainable(True, unfreeze_last_n_blocks=3)

    f = FlatMultiScaleDenseNet(pretrained=False, freeze_backbone=True)
    out = f(x)
    print("flat:", tuple(out.shape))
    assert out.shape == (2, 4)
    print("OK")
