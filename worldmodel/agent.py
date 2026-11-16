"""Agent = world model + actor-critic, with a stateful policy for acting in batched envs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .envs import Discrete
from .models.actor_critic import ActorCritic
from .models.world_model import WorldModel


class Agent(nn.Module):
    def __init__(self, obs_shape, action_space, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.discrete = isinstance(action_space, Discrete) or hasattr(action_space, "n")
        self.action_dim = int(action_space.n) if self.discrete else int(np.prod(action_space.shape))
        self.wm = WorldModel(tuple(obs_shape), self.action_dim, cfg.model)
        self.ac = ActorCritic(self.wm.rssm.feat_dim, self.action_dim, self.discrete, cfg.ac)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------- acting
    def init_policy_state(self, n: int):
        return self.wm.rssm.initial(n), torch.zeros(n, self.action_dim, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, state, reset: np.ndarray, explore: bool = True):
        """One filtering step for a batch of envs. ``reset[i]`` marks the first frame of an episode.

        Returns (env_actions, agent_actions (N, A) as stored in replay, new_state)."""
        latent, prev_action = state
        n = obs.shape[0]
        if reset.any():
            init_latent, _ = self.init_policy_state(n)
            mask = torch.as_tensor(reset, device=self.device, dtype=torch.bool)

            def blend(new, old):
                return torch.where(mask.view(-1, *[1] * (old.dim() - 1)), new, old)

            latent = {k: blend(init_latent[k], v) for k, v in latent.items()}
            prev_action = blend(torch.zeros_like(prev_action), prev_action)
        obs_t = torch.as_tensor(obs, device=self.device)
        embed = self.wm.encoder(self.wm.preprocess(obs_t))
        latent, _ = self.wm.rssm.obs_step(latent, prev_action, embed)
        dist = self.ac.policy(self.wm.rssm.feat(latent))
        action = dist.sample() if explore else dist.mode()
        action = action.detach()
        return self.to_env_actions(action), action.cpu().numpy(), (latent, action)

    def to_env_actions(self, action: torch.Tensor):
        if self.discrete:
            return action.argmax(-1).cpu().numpy()
        return action.cpu().numpy()

    def random_actions(self, n: int, rng: np.random.Generator):
        if self.discrete:
            idx = rng.integers(self.action_dim, size=n)
            return idx, np.eye(self.action_dim, dtype=np.float32)[idx]
        a = rng.uniform(-1, 1, (n, self.action_dim)).astype(np.float32)
        return a, a

    def one_hot(self, env_action) -> np.ndarray:
        if self.discrete:
            return F.one_hot(torch.as_tensor(env_action), self.action_dim).float().numpy()
        return np.asarray(env_action, np.float32)
