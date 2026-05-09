"""
losses.py
=========
Supervised Contrastive Loss — rewritten from scratch
----------------------------------------------------
Place at: R26-DS-015/shared/utils/losses.py

Author: R26-DS-015 Vision Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    For each anchor i, the loss pulls embeddings with the same label
    (positives) closer and pushes different-label embeddings (negatives)
    apart on the unit hypersphere.

    Args:
        temperature (float): Scales the similarity logits. Default: 0.07

    Input:
        features : (N, D) L2-normalised embeddings
        labels   : (N,)  integer severity labels (0, 1, 2)

    Returns:
        Scalar loss.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        N = features.shape[0]

        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ── Ensure features are L2-normalised ───────────────────────────
        features = F.normalize(features, p=2, dim=1)

        # ── Pairwise cosine similarity matrix (N x N) ───────────────────
        # Since features are L2-normalised, dot product = cosine similarity
        sim = torch.matmul(features, features.T) / self.temperature  # (N, N)

        # ── Build positive mask ──────────────────────────────────────────
        # pos_mask[i][j] = 1 if labels[i] == labels[j] AND i != j
        labels   = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (N, N)
        eye      = torch.eye(N, device=device)
        pos_mask = pos_mask * (1 - eye)   # remove diagonal (self-pairs)
        neg_mask = (1 - eye)              # all pairs except self

        # ── Check if there are any positive pairs ────────────────────────
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ── Numerical stability: subtract row max ────────────────────────
        sim_max  = sim.max(dim=1, keepdim=True).values.detach()
        sim      = sim - sim_max

        # ── For each anchor i, compute log of P(j | i) for all j ≠ i ───
        # exp_sim[i][j] = exp(sim[i][j]) for j ≠ i, else 0
        exp_sim  = torch.exp(sim) * neg_mask          # (N, N)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # ── Mean log-probability over positive pairs for each anchor ─────
        # Only consider anchors that have at least one positive
        num_pos        = pos_mask.sum(dim=1)           # (N,)
        has_pos        = num_pos > 0                   # (N,) bool

        mean_log_prob  = (pos_mask * log_prob).sum(dim=1)   # (N,)
        mean_log_prob  = mean_log_prob[has_pos] / num_pos[has_pos]

        # ── Final loss ───────────────────────────────────────────────────
        loss = -mean_log_prob.mean()
        return loss


class CombinedLoss(nn.Module):
    """
    CrossEntropy + SupCon combined loss.

        total = alpha * CE + (1 - alpha) * SupCon

    Args:
        alpha       (float): CE weight. Default: 0.5
        temperature (float): SupCon temperature. Default: 0.07
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.alpha  = alpha
        self.ce     = nn.CrossEntropyLoss()
        self.supcon = SupConLoss(temperature=temperature)

    def forward(
        self,
        embeddings:      torch.Tensor,   # (N, 256) L2-normalised z_img
        logits:          torch.Tensor,   # (N, num_classes)
        class_labels:    torch.Tensor,   # (N,) disease class (AD=0, MCI=1, CN=2)
        severity_labels: torch.Tensor,   # (N,) severity (CN=0, MCI=1, AD=2)
    ) -> tuple:
        """
        Returns (total_loss, ce_loss, supcon_loss).
        All three are scalar tensors.
        """
        ce_val = self.ce(logits, class_labels)

        supcon_val = self.supcon(embeddings, severity_labels)

        # Guard: if SupCon returned nan/inf, fall back to zero
        if not torch.isfinite(supcon_val):
            supcon_val = torch.tensor(0.0, device=logits.device,
                                      requires_grad=False)

        total = self.alpha * ce_val + (1.0 - self.alpha) * supcon_val
        return total, ce_val, supcon_val