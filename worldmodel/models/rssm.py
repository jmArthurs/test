"""Recurrent State-Space Model (DreamerV3 flavour).

State = deterministic GRU state ``h`` + stochastic categorical latent ``z`` (``stoch`` variables
with ``classes`` classes each, sampled with straight-through gradients and 1% uniform mixing).

Conventions (shared with the replay buffer): at index ``t`` a sequence stores the observation
``o_t`` and the action ``a_t`` that was taken *before* ``o_t`` was observed (zeros at episode
start). Hence ``observe`` consumes ``(embed_t, action_t)`` pairs directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

State = dict[str, torch.Tensor]


class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter: int = 512,
        stoch: int = 32,
        classes: int = 32,
        hidden: int = 512,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.deter, self.stoch, self.classes, self.unimix = deter, stoch, classes, unimix
        z_dim = stoch * classes
        self.img_in = nn.Sequential(nn.Linear(z_dim + action_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, deter)
        self.prior_net = nn.Sequential(
            nn.Linear(deter, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, z_dim)
        )
        self.post_net = nn.Sequential(
            nn.Linear(deter + embed_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, z_dim)
        )
        self.init_deter = nn.Parameter(torch.zeros(deter))

    @property
    def feat_dim(self) -> int:
        return self.deter + self.stoch * self.classes

    # ------------------------------------------------------------------ helpers
    def _logits(self, raw: torch.Tensor) -> torch.Tensor:
        logits = raw.reshape(*raw.shape[:-1], self.stoch, self.classes)
        if self.unimix:
            probs = F.softmax(logits, -1)
            probs = (1 - self.unimix) * probs + self.unimix / self.classes
            logits = probs.log()
        else:
            logits = F.log_softmax(logits, -1)
        return logits

    def _sample(self, logits: torch.Tensor, sample: bool = True) -> torch.Tensor:
        probs = logits.exp()
        if sample:
            idx = torch.multinomial(probs.reshape(-1, self.classes), 1).reshape(probs.shape[:-1])
        else:
            idx = probs.argmax(-1)
        onehot = F.one_hot(idx, self.classes).float()
        onehot = onehot + probs - probs.detach()  # straight-through
        return onehot.reshape(*onehot.shape[:-2], -1)

    def initial(self, batch_size: int) -> State:
        deter = torch.tanh(self.init_deter).expand(batch_size, -1)
        logits = self._logits(self.prior_net(deter))
        return {"deter": deter, "stoch": self._sample(logits, sample=False), "logits": logits}

    @staticmethod
    def feat(state: State) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], -1)

    # ----------------------------------------------------------------- dynamics
    def img_step(self, prev: State, action: torch.Tensor, sample: bool = True) -> State:
        """Prior transition p(h_t, z_t | h_{t-1}, z_{t-1}, a_t)."""
        x = self.img_in(torch.cat([prev["stoch"], action], -1))
        deter = self.gru(x, prev["deter"])
        logits = self._logits(self.prior_net(deter))
        return {"deter": deter, "stoch": self._sample(logits, sample), "logits": logits}

    def obs_step(self, prev: State, action: torch.Tensor, embed: torch.Tensor, sample: bool = True):
        """Posterior q(z_t | h_t, o_t). Returns ``(posterior, prior)``."""
        prior = self.img_step(prev, action, sample)
        logits = self._logits(self.post_net(torch.cat([prior["deter"], embed], -1)))
        post = {"deter": prior["deter"], "stoch": self._sample(logits, sample), "logits": logits}
        return post, prior

    def observe(self, embed: torch.Tensor, action: torch.Tensor, state: State | None = None):
        """Filter a sequence. ``embed: (B, T, E)``, ``action: (B, T, A)`` -> stacked post/prior."""
        B, T = embed.shape[:2]
        state = state if state is not None else self.initial(B)
        posts, priors = [], []
        for t in range(T):
            state, prior = self.obs_step(state, action[:, t], embed[:, t])
            posts.append(state)
            priors.append(prior)
        stack = lambda xs: {k: torch.stack([x[k] for x in xs], 1) for k in xs[0]}  # noqa: E731
        return stack(posts), stack(priors)

    def imagine(self, state: State, actions: torch.Tensor) -> State:
        """Open-loop rollout with given actions ``(B, T, A)``; returns stacked priors."""
        out = []
        for t in range(actions.shape[1]):
            state = self.img_step(state, actions[:, t])
            out.append(state)
        return {k: torch.stack([x[k] for x in out], 1) for k in out[0]}

    # --------------------------------------------------------------------- loss
    @staticmethod
    def kl(lhs_logits: torch.Tensor, rhs_logits: torch.Tensor) -> torch.Tensor:
        """KL(lhs || rhs) summed over the stochastic variables."""
        return (lhs_logits.exp() * (lhs_logits - rhs_logits)).sum((-2, -1))

    def kl_loss(self, post: State, prior: State, free_nats: float = 1.0):
        """DreamerV3 KL balancing with free bits. Returns (dyn_loss, rep_loss, raw_kl)."""
        dyn = self.kl(post["logits"].detach(), prior["logits"])
        rep = self.kl(post["logits"], prior["logits"].detach())
        return dyn.clamp(min=free_nats).mean(), rep.clamp(min=free_nats).mean(), dyn.mean().detach()
