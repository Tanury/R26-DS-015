"""
vision_encoder.py
=================
R26-DS-015 — Vision Encoder (Retinal OCT / Multiple Sclerosis)

THICKNESS-ONLY architecture (finalized after experimentation -- see
project notes): with only 35 subjects (14 HC / 21 MS), a 3D CNN branch
over raw OCT volumes was evaluated and dropped as impractical (would
badly overfit; no OCT-domain pretrained weights available, unlike the
MRI branch's MedicalNet init). Instead, this encoder operates purely on
the 8-d central-window (foveal B-scan) retinal-layer thickness vector,
which LOOCV evaluation confirmed carries real signal (Random Forest:
65.7% accuracy, 0.745 ROC-AUC) -- whole-scan averaging and engineered
layer-ratio features were also tried and did not outperform this.

Architecture:
    Input thickness (8,)  — central-window per-layer thickness features
                             in microns (see preprocessing/thickness.py,
                             preprocessing/batch_preprocess_thickness.py)

    thickness → ThicknessEncoder            → 128-d z_oct
        → VisionEncoder.shared_proj          → 256-d
        → L2 normalize → z_img   ← fusion engine input (same space as MRI z_img)
        → classifier → logits    ← training only

Note on provenance: retinal-layer thickness as an MS biomarker is a
well-established idea in the OCT/MS literature. This implementation
(preprocessing, feature engineering, and this encoder) is written from
scratch for R26-DS-015, not derived from any classification repository.

Author: R26-DS-015 Vision Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThicknessEncoder(nn.Module):
    """
    MLP encoder for the 8-d central-window retinal-layer-thickness vector.

    Args:
        in_dim  (int): Number of thickness features per sample. 8 by
                       default (central-window RNFL/GCIP/INL/OPL/ONL/
                       IS/OS/RPE, confirmed against the JHU manual
                       delineation dataset).
        out_dim (int): Output embedding dimension.
    """

    IN_DIM_DEFAULT = 8

    def __init__(self, in_dim: int = IN_DIM_DEFAULT, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(32, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, thickness: torch.Tensor) -> torch.Tensor:
        return self.net(thickness)


class VisionEncoder(nn.Module):
    """
    Full Vision Encoder for OCT-based MS risk assessment (thickness-only).

    Input:  thickness (B, 8) central-window per-layer thickness features (microns)
    Output: z_img  — (B, 256) L2-normalised embedding for fusion engine
                      (same embedding space as the MRI z_img)
            logits — (B, num_classes) for classification training
    """

    def __init__(
        self,
        num_classes: int = 2,
        thickness_dim: int = ThicknessEncoder.IN_DIM_DEFAULT,
        device: torch.device = None,
        verbose: bool = True,
    ):
        super().__init__()

        self.thickness_encoder = ThicknessEncoder(in_dim=thickness_dim, out_dim=128)

        # Shared projection: 128 -> 256-d embedding space (matches MRI branch)
        self.shared_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Classifier head (used during training, not inference)
        # num_classes=2: MS vs Healthy (JHU dataset has both)
        self.classifier = nn.Linear(256, num_classes)

        if device is not None:
            self.to(device)

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            print(f"\n  VisionEncoder (OCT/MS, thickness-only) — MLP on 8-d central-window features")
            print(f"  Embedding     : z_img (256-d, L2-normalised)")
            print(f"  Classes       : {num_classes}")
            print(f"  Parameters    : {total:,} total")

    def forward(self, thickness: torch.Tensor) -> tuple:
        z_thick = self.thickness_encoder(thickness)   # (B, 128)
        z_proj = self.shared_proj(z_thick)              # (B, 256)
        z_img = F.normalize(z_proj, p=2, dim=1)         # L2 normalise -> unit sphere
        logits = self.classifier(z_img)                 # (B, num_classes)
        return z_img, logits

    def encode(self, thickness: torch.Tensor) -> torch.Tensor:
        """Extract z_img embedding only (for fusion engine inference)."""
        z_img, _ = self.forward(thickness)
        return z_img


def build_encoder(
    num_classes: int = 2,
    thickness_dim: int = ThicknessEncoder.IN_DIM_DEFAULT,
    device: torch.device = None,
    verbose: bool = True,
) -> VisionEncoder:
    """
    Factory function for VisionEncoder (OCT/MS branch, thickness-only).

    Args:
        num_classes   : Number of output classes (2 for MS/Healthy)
        thickness_dim : Dimensionality of thickness feature vector (8 default)
        device        : torch.device to move model to
        verbose       : Print architecture summary on construction

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
        num_classes=num_classes, thickness_dim=thickness_dim, device=device, verbose=verbose
    )