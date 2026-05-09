"""
vision_encoder.py

MRI Stream Vision Encoder — R26-DS-015
--------------------------------------
Backbone: 3D ResNet-18 (MONAI)

Architecture:
    MRI Input (1, 96, 96, 96)
            ↓
    3D ResNet-18 backbone (MONAI)
    — pretrained weights optional
    — final FC layer removed
            ↓
    Global Average Pooling → 512-d feature vector
            ↓
    Modality Projection Head:
        Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.3)
        Linear(256 → 128) → BatchNorm → ReLU
            ↓
    128-d modality embedding (z_mri)
            ↓
    Shared Projection Head:
        Linear(128 → 256) → BatchNorm → ReLU → L2 Normalize
            ↓
    256-d z_img  ← forwarded to fusion engine (Aarabhi)

            ↓ (separate branch — training supervision only)
    Classification Head:
        Linear(256 → num_classes)
            ↓
    Class logits (AD / MCI / CN)

Why ResNet-18 over DenseNet121:
    - Most cited architecture in the literature review
      (Wen et al. 2020, Apurva et al. 2025 both use ResNet variants)
    - Lighter than DenseNet121 (~33M params vs ~7M — but ResNet-18 is
      the smallest ResNet variant, making it feasible on M4 Air)
    - Residual connections handle vanishing gradients well on 3D volumes
    - Clean, defensible architecture choice for dissertation writing
    - MONAI's ResNet implementation supports 3D natively with
      spatial_dims=3

Why ResNet-18 specifically (not ResNet-50):
    - ResNet-50 has ~25M params in 3D — too heavy for M4 Air with
      batch_size=2 on 96^3 volumes without running out of memory
    - ResNet-18 has ~11M params in 3D — manageable on M4 with MPS
    - For datasets of 100-300 files per class, ResNet-18 generalises
      better than deeper networks (less overfitting risk)

Note on pretrained weights:
    MONAI's 3D ResNet does not ship with MRI-pretrained weights by
    default. Set pretrained=False (default). If you later obtain
    Med3D pretrained weights (available on GitHub: Tencent/MedicalNet),
    you can load them via the resume mechanism in train.py.

Place at:
    R26-DS-015/mri/alzheimers/models/vision_encoder.py
    (will move to shared/utils/ when Parkinson's + MS are added)

Author: R26-DS-015 Vision Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet18


class MRIEncoder(nn.Module):
    """
    3D MRI encoder: ResNet-18 backbone → modality projection → 128-d z_mri.

    Args:
        freeze_backbone (bool): Freeze ResNet-18 weights during Phase 1 training.
                                Set True for first 20 epochs, then unfreeze.
                                Default: False
    """

    # ResNet-18 output feature dimension after global average pooling
    BACKBONE_DIM = 512

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()

        # ── ResNet-18 backbone ──────────────────────────────────────────
        # spatial_dims=3  → 3D convolutions for volumetric MRI
        # n_input_channels=1 → grayscale MRI (single channel)
        # num_classes=1   → placeholder; we remove the FC head below
        # feed_forward=False → returns feature maps before final FC
        #                      (MONAI ResNet supports this flag)
        self._backbone = resnet18(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=self.BACKBONE_DIM,
            feed_forward=False,   # return pooled features, not logits
        )

        if freeze_backbone:
            for param in self._backbone.parameters():
                param.requires_grad = False
            print("  ℹ️   Backbone frozen (Phase 1 training)")

        # ── Modality Projection Head ────────────────────────────────────
        # 512-d backbone features → 128-d z_mri
        # Two-layer MLP with BatchNorm and Dropout for regularisation
        self.modality_proj = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, 1, 96, 96, 96) preprocessed MRI volume

        Returns:
            z_mri : (batch, 128) modality-specific embedding
        """
        features = self._backbone(x)    # (batch, 512)
        z_mri    = self.modality_proj(features)   # (batch, 128)
        return z_mri

    def unfreeze(self):
        """Unfreeze backbone for Phase 2 fine-tuning."""
        for param in self._backbone.parameters():
            param.requires_grad = True
        print("  ✅  Backbone unfrozen (Phase 2 fine-tuning)")


class VisionEncoder(nn.Module):
    """
    Full Vision Encoder: MRI stream → shared projection → 256-d z_img.

    Forward pass returns:
        z_img  : (batch, 256) L2-normalized — for SupCon loss + fusion engine
        logits : (batch, num_classes)       — for CrossEntropy + evaluation metrics

    The classification head gives you accuracy/F1/AUC for your supervisor.
    The z_img is what your team's fusion engine (Aarabhi) consumes.

    After training, only z_img is used in production.
    The classification head is kept in the checkpoint but not called
    by embed.py or the fusion engine.

    Args:
        num_classes     (int):  3 for AD/MCI/CN. Default: 3
        freeze_backbone (bool): Freeze backbone for Phase 1. Default: False
    """

    def __init__(
        self,
        num_classes:     int  = 3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes

        # ── MRI Stream ──────────────────────────────────────────────────
        self.mri_encoder = MRIEncoder(freeze_backbone=freeze_backbone)

        # ── Shared Projection Head ──────────────────────────────────────
        # 128-d z_mri → 256-d z_img (L2 normalized)
        # Matches proposal FR5:
        #   FC → BatchNorm → ReLU → L2 Normalize
        self.shared_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # ── Classification Head ─────────────────────────────────────────
        # Provides supervised signal during training.
        # Gives accuracy/F1/AUC for evaluation.
        # Not used by the fusion engine.
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x : (batch, 1, 96, 96, 96) preprocessed MRI volume

        Returns:
            z_img  : (batch, 256) L2-normalized embedding
            logits : (batch, num_classes) classification scores
        """
        # MRI stream → 128-d modality embedding
        z_mri = self.mri_encoder(x)

        # Shared projection → 256-d
        z_proj = self.shared_proj(z_mri)

        # L2 normalize onto unit hypersphere
        # Required for: SupCon loss (cosine similarity) + fusion engine
        z_img = F.normalize(z_proj, p=2, dim=1)

        # Classification logits (training + evaluation only)
        logits = self.classifier(z_img)

        return z_img, logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns only z_img. Used by embed.py and the fusion engine.

        Args:
            x : (batch, 1, 96, 96, 96)

        Returns:
            z_img : (batch, 256) L2-normalized
        """
        z_img, _ = self.forward(x)
        return z_img

    def unfreeze_backbone(self):
        """
        Call this at the start of Phase 2 training to unfreeze the
        ResNet-18 backbone for full fine-tuning.

        Usage in train.py:
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        """
        self.mri_encoder.unfreeze()

    def get_embedding_dim(self) -> int:
        """Always 256."""
        return 256


def build_encoder(
    num_classes:     int           = 3,
    freeze_backbone: bool          = False,
    device:          torch.device  = None,
) -> VisionEncoder:
    """
    Factory function — builds and returns a VisionEncoder.

    Args:
        num_classes     : 3 for AD/MCI/CN (Alzheimer's)
                          2 for PD/HC (Parkinson's) — set when reusing for PD
        freeze_backbone : True for Phase 1, False for Phase 2
        device          : torch.device to move model to

    Returns:
        VisionEncoder instance on the specified device
    """
    model = VisionEncoder(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )

    if device is not None:
        model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"  VisionEncoder — 3D ResNet-18 backbone")
    print(f"  Embedding     : z_img (256-d, L2-normalized)")
    print(f"  Classes       : {num_classes}")
    print(f"  Parameters    : {total_params:,} total  |  "
          f"{trainable_params:,} trainable")

    return model