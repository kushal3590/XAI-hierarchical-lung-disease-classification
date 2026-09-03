"""
Multi-scale feature extraction + hierarchical classification model.

Implements O3 (multi-scale deep learning model) and O4 (hierarchical
multilevel classification) from the Phase 1 report.

Architecture:
  EfficientNet-B0 backbone (pretrained on ImageNet)
    -> tap 3 intermediate feature maps (early / mid / deep blocks)
    -> Global Average Pool each independently
    -> concatenate into one multi-scale feature vector
    -> shared FC layer
    -> two classification heads:
         stage1_head: Normal (0) vs Abnormal (1)
         stage2_head: Pneumonia (0) / Tuberculosis (1) / COVID-19 (2)
                      (only meaningful/trained when stage1 == Abnormal;
                       handled via CrossEntropyLoss(ignore_index=-1))

Note on report alignment: the Phase 1 report specifies a 3-stage hierarchy
(designed for a 17-class problem). Since our merged dataset has only 4
classes total (Normal + 3 diseases), a 3rd stage has nothing left to
subdivide, so this is a 2-stage hierarchy — see chat discussion. This is
consistent with your own Phase 1 research-gap table (G2 marked "Partially"
addressed).

Tap points (features[2]/[5]/[8] of torchvision's efficientnet_b0) are chosen
to approximate "early / mid / deep" as described in the report's Figure 2,
though exact torchvision block indexing differs slightly from the generic
4-conv-block diagram in the report — channel dimensions are read dynamically
at init time (a dummy forward pass) rather than hardcoded, so this is robust
to minor torchvision version differences.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class MultiScaleEfficientNet(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.backbone = backbone.features  # nn.Sequential of stem + 7 MBConv stages + final conv

        # Tap indices chosen to span fine-grained -> mid -> semantic features
        #   features[2]: early MBConv stage (fine-grained: edges, texture)
        #   features[5]: mid MBConv stage (intermediate: shapes, local patterns)
        #   features[8]: final conv (deep/semantic: disease-level patterns)
        self.tap_indices = {2: "early", 5: "mid", 8: "deep"}
        self._features = {}
        for idx, name in self.tap_indices.items():
            self.backbone[idx].register_forward_hook(self._make_hook(name))

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Determine actual channel dims via a dummy forward pass — avoids
        # hardcoding numbers that could silently drift across torchvision versions
        with torch.no_grad():
            self.eval()
            dummy = torch.zeros(1, 3, 224, 224)
            self._features = {}
            _ = self.backbone(dummy)
            concat_dim = sum(self._features[name].shape[1] for name in self.tap_indices.values())
        self.train()

        self.shared = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.stage1_head = nn.Linear(256, 2)  # Normal vs Abnormal
        self.stage2_head = nn.Linear(256, 3)  # Pneumonia / TB / COVID-19

        print(f"[MultiScaleEfficientNet] Concatenated multi-scale feature dim: {concat_dim}")

        if freeze_backbone:
            self.set_backbone_trainable(False)

    def _make_hook(self, name):
        def hook(module, inp, out):
            self._features[name] = out
        return hook

    def set_backbone_trainable(self, trainable: bool, unfreeze_last_n_blocks: int = None):
        """Freeze/unfreeze the backbone. If unfreeze_last_n_blocks is set,
        only the last N children of self.backbone get unfrozen (progressive
        fine-tuning) rather than the whole thing — gentler on a 6GB GPU and
        generally trains more stably."""
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
        print(f"[MultiScaleEfficientNet] Backbone trainable params: {n_trainable:,} / {n_total:,}")

    def forward(self, x):
        self._features = {}
        _ = self.backbone(x)  # populates self._features via hooks

        pooled = [self.gap(self._features[name]).flatten(1) for name in self.tap_indices.values()]
        feat = torch.cat(pooled, dim=1)

        shared = self.shared(feat)
        stage1_logits = self.stage1_head(shared)
        stage2_logits = self.stage2_head(shared)
        return stage1_logits, stage2_logits


def combine_predictions(stage1_logits, stage2_logits):
    """Combine the two heads into a single 4-class prediction (0=Normal,
    1=Pneumonia, 2=Tuberculosis, 3=COVID-19) for reporting overall accuracy
    comparable to the flat-classification numbers in the literature review."""
    stage1_pred = stage1_logits.argmax(dim=1)  # 0=Normal, 1=Abnormal
    stage2_pred = stage2_logits.argmax(dim=1)  # 0=Pneumonia,1=TB,2=COVID-19

    # Map: if Normal -> flat class 0; else flat class = stage2_pred + 1
    flat_pred = torch.where(
        stage1_pred == 0,
        torch.zeros_like(stage2_pred),
        stage2_pred + 1,
    )
    return flat_pred


def combine_predictions_thresholded(stage1_logits, stage2_logits, abnormal_threshold: float = 0.5):
    """Same as combine_predictions(), but Stage 1's Normal/Abnormal decision
    uses a tunable probability threshold on P(Abnormal) instead of the
    default argmax (equivalent to threshold=0.5).

    Per-class AUC on this project's model is high across all 4 classes
    (0.98+), which means the model already ranks Abnormal-vs-Normal well —
    the issue is the OPERATING POINT: at the default 0.5 cutoff, Pneumonia
    recall specifically suffers (many true Pneumonia cases score as
    "Abnormal" with moderate-but-sub-0.5 confidence and get gated to
    Normal). Lowering the threshold trades some Normal precision for higher
    Pneumonia (and overall) recall — this function makes that trade-off
    tunable without retraining. See scripts/tune_stage1_threshold.py.
    """
    stage1_probs = torch.softmax(stage1_logits, dim=1)
    p_abnormal = stage1_probs[:, 1]
    stage1_pred = (p_abnormal >= abnormal_threshold).long()  # 0=Normal, 1=Abnormal

    stage2_pred = stage2_logits.argmax(dim=1)
    flat_pred = torch.where(
        stage1_pred == 0,
        torch.zeros_like(stage2_pred),
        stage2_pred + 1,
    )
    return flat_pred


class FlatMultiScaleEfficientNet(nn.Module):
    """Baseline for comparison: identical multi-scale EfficientNet-B0 backbone
    and tap points as MultiScaleEfficientNet, but a single flat 4-class head
    (Normal / Pneumonia / Tuberculosis / COVID-19) instead of the two
    hierarchical stages. Isolates the hierarchical-vs-flat variable — same
    data, same backbone, same training schedule, only the classification
    head structure differs. This is what your report's Comparative Analysis
    / Expected Contributions section needs to actually demonstrate the
    hierarchical approach's benefit, rather than just citing other papers'
    numbers."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True, num_classes: int = 4):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.backbone = backbone.features

        self.tap_indices = {2: "early", 5: "mid", 8: "deep"}
        self._features = {}
        for idx, name in self.tap_indices.items():
            self.backbone[idx].register_forward_hook(self._make_hook(name))

        self.gap = nn.AdaptiveAvgPool2d(1)

        with torch.no_grad():
            self.eval()
            dummy = torch.zeros(1, 3, 224, 224)
            self._features = {}
            _ = self.backbone(dummy)
            concat_dim = sum(self._features[name].shape[1] for name in self.tap_indices.values())
        self.train()

        self.shared = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, num_classes)  # single flat head

        print(f"[FlatMultiScaleEfficientNet] Concatenated multi-scale feature dim: {concat_dim}")

        if freeze_backbone:
            self.set_backbone_trainable(False)

    def _make_hook(self, name):
        def hook(module, inp, out):
            self._features[name] = out
        return hook

    def set_backbone_trainable(self, trainable: bool, unfreeze_last_n_blocks: int = None):
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
        print(f"[FlatMultiScaleEfficientNet] Backbone trainable params: {n_trainable:,} / {n_total:,}")

    def forward(self, x):
        self._features = {}
        _ = self.backbone(x)
        pooled = [self.gap(self._features[name]).flatten(1) for name in self.tap_indices.values()]
        feat = torch.cat(pooled, dim=1)
        shared = self.shared(feat)
        return self.classifier(shared)


if __name__ == "__main__":
    # Quick standalone check: python src/model.py
    model = MultiScaleEfficientNet(pretrained=True, freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    s1, s2 = model(x)
    print("stage1_logits:", s1.shape)
    print("stage2_logits:", s2.shape)
    print("combined flat prediction:", combine_predictions(s1, s2))
