"""Actor-critic trained purely in imagination (DreamerV3-style).

* Imagined rollouts of length ``horizon`` start from every posterior state of the WM batch.
* Critic regresses symlog lambda-returns, regularised toward a slow EMA copy.
* Actor uses REINFORCE with returns normalised by a running 5-95th percentile range,
  plus an entropy bonus. The world model is frozen during this phase.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ..config import ActorCriticConfig
from .networks import MLP, BoundedNormal, OneHotCategorical, symexp, symlog


class ReturnNormalizer(nn.Module):
    def __init__(self, decay: float = 0.99, low: float = 0.05, high: float = 0.95):
        super().__init__()
        self.decay, self.low, self.high = decay, low, high
        self.register_buffer("lo", torch.zeros(()))
        self.register_buffer("hi", torch.zeros(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten().float()
        lo, hi = torch.quantile(x, self.low), torch.quantile(x, self.high)
        if not self.initialized:
            self.lo.copy_(lo)
            self.hi.copy_(hi)
            self.initialized.fill_(True)
        else:
            self.lo.lerp_(lo, 1 - self.decay)
            self.hi.lerp_(hi, 1 - self.decay)
        return torch.clamp(self.hi - self.lo, min=1.0)


def lambda_return(reward, value, discount, lam):
    """``reward, discount: (H, N)`` for transitions t->t+1, ``value: (H+1, N)``."""
    H = reward.shape[0]
    out = []
    nxt = value[-1]
    for t in reversed(range(H)):
        nxt = reward[t] + discount[t] * ((1 - lam) * value[t + 1] + lam * nxt)
        out.append(nxt)
    return torch.stack(out[::-1])


class ActorCritic(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, discrete: bool, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        self.discrete = discrete
        self.action_dim = action_dim
        self.actor = MLP(feat_dim, action_dim if discrete else 2 * action_dim, cfg.hidden, cfg.layers)
        self.critic = MLP(feat_dim, 1, cfg.hidden, cfg.layers, zero_init=True)
        self.slow_critic = copy.deepcopy(self.critic)
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        self.normalizer = ReturnNormalizer()

    def policy(self, feat: torch.Tensor):
        out = self.actor(feat)
        return OneHotCategorical(out) if self.discrete else BoundedNormal(out)

    @torch.no_grad()
    def update_slow_critic(self) -> None:
        for ps, p in zip(self.slow_critic.parameters(), self.critic.parameters()):
            ps.lerp_(p, self.cfg.slow_critic_tau)

    def loss(self, world_model, post: dict[str, torch.Tensor]):
        cfg = self.cfg
        rssm = world_model.rssm
        state = {k: v.detach().reshape(-1, *v.shape[2:]) for k, v in post.items()}

        with torch.no_grad():
            feats, actions = [rssm.feat(state)], []
            for _ in range(cfg.horizon):
                a = self.policy(feats[-1]).sample()
                state = rssm.img_step(state, a)
                feats.append(rssm.feat(state))
                actions.append(a)
            feats = torch.stack(feats)  # (H+1, N, F)
            actions = torch.stack(actions)  # (H, N, A)
            reward = symexp(world_model.reward_head(feats[1:]).squeeze(-1))
            cont = torch.sigmoid(world_model.cont_head(feats[1:]).squeeze(-1))
            value = symexp(self.critic(feats).squeeze(-1))
            returns = lambda_return(reward, value, cfg.gamma * cont, cfg.lam)
            weight = torch.cumprod(torch.cat([torch.ones_like(cont[:1]), cont[:-1]]), 0)
            scale = self.normalizer.update(returns)
            adv = (returns - value[:-1]) / scale

        dist = self.policy(feats[:-1])
        entropy = dist.entropy()
        actor_loss = -(weight * (dist.log_prob(actions) * adv + cfg.entropy * entropy)).mean()

        v_pred = self.critic(feats[:-1]).squeeze(-1)
        with torch.no_grad():
            slow = self.slow_critic(feats[:-1]).squeeze(-1)
        critic_loss = (
            weight * ((v_pred - symlog(returns)) ** 2 + cfg.slow_critic_reg * (v_pred - slow) ** 2)
        ).mean()

        metrics = {
            "ac/actor_loss": actor_loss.item(),
            "ac/critic_loss": critic_loss.item(),
            "ac/entropy": entropy.mean().item(),
            "ac/imag_return": returns.mean().item(),
            "ac/imag_reward": reward.mean().item(),
            "ac/return_scale": scale.item(),
        }
        return actor_loss, critic_loss, metrics
