"""Shared building blocks: MLPs, conv encoder/decoder, symlog transforms, action distributions."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


class MLP(nn.Module):
    """Linear -> LayerNorm -> SiLU blocks followed by a linear output layer."""

    def __init__(self, inp: int, out: int, hidden: int = 512, layers: int = 2, zero_init: bool = False):
        super().__init__()
        mods: list[nn.Module] = []
        d = inp
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d = hidden
        self.out = nn.Linear(d, out)
        if zero_init:  # DreamerV3 trick for reward/value heads: start predicting 0
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        self.body = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


def conv_stages(image_size: int) -> int:
    """Number of stride-2 conv stages to reach a 4x4 feature map."""
    stages = int(round(math.log2(image_size))) - 2
    if 2 ** (stages + 2) != image_size:
        raise ValueError(f"image size must be a power of two >= 8, got {image_size}")
    return stages


class ConvEncoder(nn.Module):
    """Images ``(..., H, W, C)`` in [-0.5, 0.5] -> flat embeddings ``(..., out_dim)``."""

    def __init__(self, image_size: int = 64, channels: int = 3, depth: int = 32):
        super().__init__()
        stages = conv_stages(image_size)
        layers: list[nn.Module] = []
        c_in, c = channels, depth
        for _ in range(stages):
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.GroupNorm(1, c), nn.SiLU()]
            c_in, c = c, c * 2
        self.net = nn.Sequential(*layers, nn.Flatten())
        self.out_channels = c_in
        self.out_dim = c_in * 4 * 4

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:]).permute(0, 3, 1, 2)
        return self.net(x).reshape(*batch, -1)


class ConvDecoder(nn.Module):
    """Latent features ``(..., F)`` -> images ``(..., H, W, C)`` (mean of a unit Gaussian)."""

    def __init__(self, feat_dim: int, image_size: int = 64, channels: int = 3, depth: int = 32):
        super().__init__()
        stages = conv_stages(image_size)
        c = depth * 2 ** (stages - 1)
        self.c0 = c
        self.fc = nn.Linear(feat_dim, c * 4 * 4)
        layers: list[nn.Module] = []
        for _ in range(stages - 1):
            layers += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.GroupNorm(1, c // 2), nn.SiLU()]
            c //= 2
        layers.append(nn.ConvTranspose2d(c, channels, 4, 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        batch = feat.shape[:-1]
        x = self.fc(feat.reshape(-1, feat.shape[-1])).reshape(-1, self.c0, 4, 4)
        x = self.net(x).permute(0, 2, 3, 1)
        return x.reshape(*batch, *x.shape[1:])


# --------------------------------------------------------------------------- distributions
class OneHotCategorical:
    """Categorical over one-hot actions with uniform mixing and straight-through samples."""

    def __init__(self, logits: torch.Tensor, unimix: float = 0.01):
        probs = F.softmax(logits, -1)
        if unimix:
            probs = (1 - unimix) * probs + unimix / probs.shape[-1]
        self.probs = probs
        self.logits = probs.log()

    def sample(self) -> torch.Tensor:
        idx = torch.multinomial(self.probs.reshape(-1, self.probs.shape[-1]), 1)
        onehot = F.one_hot(idx.squeeze(-1), self.probs.shape[-1]).reshape(self.probs.shape).float()
        return onehot + self.probs - self.probs.detach()

    def mode(self) -> torch.Tensor:
        return F.one_hot(self.probs.argmax(-1), self.probs.shape[-1]).float()

    def log_prob(self, onehot: torch.Tensor) -> torch.Tensor:
        return (onehot.detach() * self.logits).sum(-1)

    def entropy(self) -> torch.Tensor:
        return -(self.probs * self.logits).sum(-1)


class BoundedNormal:
    """Diagonal Gaussian with tanh-squashed mean and bounded std, samples clipped to [-1, 1]."""

    def __init__(self, out: torch.Tensor, min_std: float = 0.1, max_std: float = 1.0):
        mean, std = out.chunk(2, -1)
        self.mean = torch.tanh(mean)
        std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
        self.dist = td.Independent(td.Normal(self.mean, std), 1)

    def sample(self) -> torch.Tensor:
        return self.dist.rsample().clamp(-1, 1)

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(a.detach())

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()
