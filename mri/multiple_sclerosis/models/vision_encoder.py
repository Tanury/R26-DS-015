"""
vision_encoder.py for MS MRI branch  

Architecture, 
built from scratch on MindGlide (MS-PINPOINT) region-volume features,
NOT a 3D CNN over raw volumes -- MindGlide already does the heavy
segmentation work (pretrained, published, zero-preprocessing), so this
encoder operates on its OUTPUT (region volumes), not on raw scans.

Feature set: 5 literature-motivated regions, normalized by each
subject's total brain volume (removes the brain-size confound found
between the two source cohorts -- see build_multifeature_classifier.py):
    Lesion, Lateral_ventricle, DGM, Corpus_callosum, White_matter
(all as fraction of total brain volume)

This combination was chosen over two more complex alternatives (all 19
raw region volumes; all 19 normalized) because it matched or exceeded
their LOOCV performance while being smaller, more clinically
interpretable, and explicitly confound-controlled -- not because it
was the single highest-scoring option tried.

Input  (5,)   -- normalized region-volume fractions
    -> RegionVolumeEncoder (MLP)      -> 128-d z_mrims
    -> VisionEncoder.shared_proj       -> 256-d
    -> L2 normalize -> z_img   <- fusion engine input (same space as
                                  the eye/ms/ OCT branch and the
                                  original AD/PD MRI branches)
    -> classifier -> logits    <- training only

Author: R26-DS-015 Vision Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionVolumeEncoder(nn.Module):
    """
    MLP encoder for the 5-d normalized region-volume feature vector.

    Args:
        in_dim  (int): Number of features per sample. 5 by default
                       (Lesion, Lateral_ventricle, DGM, Corpus_callosum,
                       White_matter, each as a fraction of total brain
                       volume).
        out_dim (int): Output embedding dimension.
    """

    IN_DIM_DEFAULT = 5

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class VisionEncoder(nn.Module):
    """
    Full Vision Encoder for MRI-based MS risk assessment
    (MindGlide-region-volume-based, not a raw-volume CNN).

    Input:  features (B, 5) normalized region-volume fractions
    Output: z_img  -- (B, 256) L2-normalised embedding for fusion engine
                       (same embedding space as the eye/ms/ OCT branch
                       and the AD/PD MRI branches)
            logits -- (B, num_classes) for classification training
    """

    def __init__(
        self,
        num_classes: int = 2,
        feature_dim: int = RegionVolumeEncoder.IN_DIM_DEFAULT,
        device: torch.device = None,
        verbose: bool = True,
    ):
        super().__init__()

        self.region_encoder = RegionVolumeEncoder(in_dim=feature_dim, out_dim=128)

        # Shared projection: 128 -> 256-d embedding space (matches every other branch)
        self.shared_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Classifier head (used during training, not inference)
        # num_classes=2: MS vs Healthy
        self.classifier = nn.Linear(256, num_classes)

        if device is not None:
            self.to(device)

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            print(f"\n  VisionEncoder (MRI/MS, MindGlide-region-based) — MLP on 5-d normalized features")
            print(f"  Embedding     : z_img (256-d, L2-normalised)")
            print(f"  Classes       : {num_classes}")
            print(f"  Parameters    : {total:,} total")

    def forward(self, features: torch.Tensor) -> tuple:
        z_region = self.region_encoder(features)   # (B, 128)
        z_proj = self.shared_proj(z_region)          # (B, 256)
        z_img = F.normalize(z_proj, p=2, dim=1)       # L2 normalise -> unit sphere
        logits = self.classifier(z_img)               # (B, num_classes)
        return z_img, logits

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Extract z_img embedding only (for fusion engine inference)."""
        z_img, _ = self.forward(features)
        return z_img


def build_encoder(
    num_classes: int = 2,
    feature_dim: int = RegionVolumeEncoder.IN_DIM_DEFAULT,
    device: torch.device = None,
    verbose: bool = True,
) -> VisionEncoder:
    """
    Factory function for VisionEncoder (MRI/MS branch, MindGlide-region-based).

    Args:
        num_classes  : Number of output classes (2 for MS/Healthy)
        feature_dim  : Dimensionality of the input feature vector (5 default)
        device       : torch.device to move model to
        verbose      : Print architecture summary on construction

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
        num_classes=num_classes, feature_dim=feature_dim, device=device, verbose=verbose
    )