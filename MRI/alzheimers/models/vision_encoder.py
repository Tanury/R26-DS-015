"""
vision_encoder.py
=================
R26-DS-015 — Vision Encoder (Brain MRI)
3D ResNet-18 backbone + MedicalNet pretrained weights support

Architecture:
    Input (1, 96, 96, 96)
    → ResNet-18 backbone (3D, feed_forward=False) → 512-d
    → MRIEncoder.modality_proj → 128-d z_mri
    → VisionEncoder.shared_proj → 256-d
    → L2 normalize → z_img  ← fusion engine input
    → classifier → logits   ← training only

Author: R26-DS-015 Vision Encoder
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet18


class MRIEncoder(nn.Module):
    """
    3D ResNet-18 backbone with optional MedicalNet pretrained weights.

    Args:
        freeze_backbone  (bool): Freeze ResNet-18 weights (Phase 1 training)
        pretrained_path  (str):  Path to MedicalNet resnet_18.pth file.
                                 If None or file missing, trains from scratch.
    """

    BACKBONE_DIM = 512

    def __init__(
        self,
        freeze_backbone: bool = False,
        pretrained_path: str = None,
    ):
        super().__init__()

        # ── 3D ResNet-18 backbone ─────────────────────────────────────────
        self._backbone = resnet18(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=self.BACKBONE_DIM,
            feed_forward=False,
        )

        # ── Load MedicalNet pretrained weights ────────────────────────────
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_medicalnet(pretrained_path)
        else:
            if pretrained_path:
                print(f"  ⚠️  MedicalNet weights not found at {pretrained_path}")
                print(f"       Training backbone from scratch.")
            else:
                print(f"  ℹ️   No pretrained weights specified — training from scratch.")

        # ── Freeze backbone for Phase 1 ───────────────────────────────────
        if freeze_backbone:
            for p in self._backbone.parameters():
                p.requires_grad = False
            print("  Backbone frozen (Phase 1 — only projection head trains)")

        # ── Projection head: 512 → 128 ────────────────────────────────────
        self.modality_proj = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

    def _load_medicalnet(self, path: str) -> None:
        """
        Load MedicalNet ResNet-18 pretrained weights.

        MedicalNet stores weights as:
            {'state_dict': {'module.layer1...': tensor, ...}}
        or directly as a flat state dict.

        Key differences from MONAI ResNet-18:
            - MedicalNet keys may have 'module.' prefix (DataParallel)
            - MedicalNet conv1 is (64, 1, 7, 7, 7) — matches our n_input_channels=1
            - Layer names match MONAI's ResNet implementation
        """
        print(f"  Loading MedicalNet weights from: {path}")

        checkpoint = torch.load(path, map_location="cpu")

        # Extract state dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        else:
            sd = checkpoint

        # Strip 'module.' prefix from DataParallel training
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

        # Match against current model
        model_sd   = self._backbone.state_dict()
        matched    = {}
        mismatched = []
        skipped    = []

        for k, v in sd.items():
            if k in model_sd:
                if v.shape == model_sd[k].shape:
                    matched[k] = v
                else:
                    mismatched.append(f"{k}: ckpt {v.shape} vs model {model_sd[k].shape}")
            else:
                skipped.append(k)

        # Load matched weights
        model_sd.update(matched)
        self._backbone.load_state_dict(model_sd)

        print(f"  ✅ MedicalNet: loaded {len(matched)}/{len(model_sd)} layers")
        if mismatched:
            print(f"     Shape mismatches ({len(mismatched)}): {mismatched[:3]}")
        if skipped:
            print(f"     Skipped {len(skipped)} keys not in model")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modality_proj(self._backbone(x))

    def unfreeze(self) -> None:
        for p in self._backbone.parameters():
            p.requires_grad = True
        print("  Backbone unfrozen (Phase 2 — full fine-tuning)")


class VisionEncoder(nn.Module):
    """
    Full Vision Encoder for neurological risk assessment.

    Input:  (B, 1, 96, 96, 96) preprocessed brain MRI
    Output: z_img  — (B, 256) L2-normalised embedding for fusion engine
            logits — (B, num_classes) for classification training
    """

    def __init__(
        self,
        num_classes: int = 3,
        freeze_backbone: bool = False,
        pretrained_path: str = None,
        device: torch.device = None,
    ):
        super().__init__()

        self.mri_encoder = MRIEncoder(
            freeze_backbone=freeze_backbone,
            pretrained_path=pretrained_path,
        )

        # Shared projection: 128 → 256-d embedding space
        self.shared_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Classifier head (used during training, not inference)
        self.classifier = nn.Linear(256, num_classes)

        if device is not None:
            self.to(device)

        total  = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"\n  VisionEncoder — 3D ResNet-18 + MedicalNet init")
        print(f"  Embedding     : z_img (256-d, L2-normalised)")
        print(f"  Classes       : {num_classes}")
        print(f"  Parameters    : {total:,} total  |  {total - frozen:,} trainable")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_mri  = self.mri_encoder(x)           # (B, 128)
        z_proj = self.shared_proj(z_mri)        # (B, 256)
        z_img  = F.normalize(z_proj, p=2, dim=1)  # L2 normalise → unit sphere
        logits = self.classifier(z_img)         # (B, num_classes)
        return z_img, logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract z_img embedding only (for fusion engine inference)."""
        z_img, _ = self.forward(x)
        return z_img

    def unfreeze_backbone(self) -> None:
        self.mri_encoder.unfreeze()


def build_encoder(
    num_classes: int = 3,
    freeze_backbone: bool = False,
    pretrained_path: str = None,
    device: torch.device = None,
) -> VisionEncoder:
    """
    Factory function for VisionEncoder.

    Args:
        num_classes      : Number of output classes (3 for AD/MCI/CN)
        freeze_backbone  : Freeze ResNet-18 for Phase 1 training
        pretrained_path  : Path to MedicalNet resnet_18.pth
        device           : torch.device to move model to

    Returns:
        VisionEncoder instance
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    return VisionEncoder(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        pretrained_path=pretrained_path,
        device=device,
    )