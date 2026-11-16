"""JEPA components: EMA target encoder, latent prediction losses, anti-collapse regularisers.

The JEPA objective replaces pixel reconstruction with *prediction in representation space*:
the world model's (imagined) latent state must predict the embedding that a slowly-moving
EMA *target* encoder assigns to the future observation. Nothing is ever decoded to pixels.

Collapse (constant embeddings make prediction trivial) is prevented by the combination of
(1) stop-gradient + EMA target network (BYOL / I-JEPA / V-JEPA), and
(2) VICReg variance/covariance terms on the online projections (as in VICReg / PLDM).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMATarget(nn.Module):
    """Frozen copy of ``online`` modules updated with an exponential moving average."""

    def __init__(self, *online: nn.Module):
        super().__init__()
        self.target = nn.ModuleList([copy.deepcopy(m) for m in online])
        for p in self.target.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.target:
            x = m(x)
        return x

    @torch.no_grad()
    def update(self, online: list[nn.Module], tau: float) -> None:
        for tgt, src in zip(self.target, online):
            for pt, ps in zip(tgt.parameters(), src.parameters()):
                pt.lerp_(ps, 1.0 - tau)
            for bt, bs in zip(tgt.buffers(), src.buffers()):
                bt.copy_(bs)


def prediction_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "cosine") -> torch.Tensor:
    """Distance between predicted and target embeddings, averaged over all leading dims."""
    target = target.detach()
    if kind == "cosine":  # BYOL-style normalised MSE, in [0, 4]
        return (2 - 2 * F.cosine_similarity(pred, target, dim=-1)).mean()
    if kind == "l1":  # I-JEPA / V-JEPA style, with layer-normalised targets
        return F.smooth_l1_loss(pred, F.layer_norm(target, target.shape[-1:]))
    if kind == "mse":
        return F.mse_loss(pred, F.layer_norm(target, target.shape[-1:]))
    raise ValueError(f"unknown jepa loss {kind!r}")


def vicreg_terms(z: torch.Tensor, eps: float = 1e-4):
    """Variance (hinge on per-dim std >= 1) and covariance (off-diagonal) penalties.

    ``z``: (N, D) batch of embeddings. Returns (variance_loss, covariance_loss).
    """
    z = z.reshape(-1, z.shape[-1]).float()
    n, d = z.shape
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / d
    return var_loss, cov_loss
