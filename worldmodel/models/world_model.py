"""World model = conv encoder + RSSM + reward/continue heads + a representation objective.

Objectives (``cfg.objective``):

* ``dreamer`` - posterior latents must reconstruct pixels (DreamerV3).
* ``jepa``    - no pixel loss: imagined latents must predict EMA-target embeddings of future
                frames, k = 1..H steps open-loop (JEPA-style latent prediction), plus
                VICReg regularisation against collapse.
* ``hybrid``  - both.

In every mode the RSSM is trained with the KL-balanced prior/posterior loss and reward/continue
prediction, so the actor-critic can be trained in imagination regardless of the objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .jepa import EMATarget, prediction_loss, vicreg_terms
from .networks import MLP, ConvDecoder, ConvEncoder, symlog
from .rssm import RSSM


class WorldModel(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], action_dim: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        size, _, channels = obs_shape
        self.encoder = ConvEncoder(size, channels, cfg.cnn_depth)
        self.rssm = RSSM(
            self.encoder.out_dim, action_dim, cfg.deter, cfg.stoch, cfg.classes, cfg.hidden, cfg.unimix
        )
        feat = self.rssm.feat_dim
        self.reward_head = MLP(feat, 1, cfg.hidden, cfg.mlp_layers, zero_init=True)
        self.cont_head = MLP(feat, 1, cfg.hidden, cfg.mlp_layers)

        self.use_recon = cfg.objective in ("dreamer", "hybrid")
        self.use_jepa = cfg.objective in ("jepa", "hybrid")
        self.decoder = (
            ConvDecoder(feat, size, channels, cfg.cnn_depth) if (self.use_recon or cfg.decoder_probe) else None
        )
        if self.use_jepa:
            self.projector = MLP(self.encoder.out_dim, cfg.jepa_dim, cfg.hidden, 1)
            self.predictor = MLP(feat, cfg.jepa_dim, cfg.hidden, cfg.mlp_layers)
            self.target = EMATarget(self.encoder, self.projector)

    # ----------------------------------------------------------------- utils
    @staticmethod
    def preprocess(obs: torch.Tensor) -> torch.Tensor:
        return obs.float() / 255.0 - 0.5

    def update_target(self) -> None:
        if self.use_jepa:
            self.target.update([self.encoder, self.projector], self.cfg.ema_tau)

    # ------------------------------------------------------------------ loss
    def loss(self, batch: dict[str, torch.Tensor]):
        """``batch`` holds ``obs (B,T,H,W,C) uint8``, ``action (B,T,A)``, ``reward (B,T)``,
        ``terminal (B,T)``. Returns ``(total_loss, metrics, posterior_states)``."""
        cfg = self.cfg
        obs = self.preprocess(batch["obs"])
        action = batch["action"]
        B, T = obs.shape[:2]

        embed = self.encoder(obs)
        post, prior = self.rssm.observe(embed, action)
        feat = self.rssm.feat(post)

        losses: dict[str, torch.Tensor] = {}
        dyn, rep, kl = self.rssm.kl_loss(post, prior, cfg.free_nats)
        losses["kl_dyn"] = cfg.kl_dyn * dyn
        losses["kl_rep"] = cfg.kl_rep * rep
        reward_pred = self.reward_head(feat).squeeze(-1)
        losses["reward"] = cfg.reward_scale * F.mse_loss(reward_pred, symlog(batch["reward"]))
        cont_logits = self.cont_head(feat).squeeze(-1)
        losses["cont"] = cfg.cont_scale * F.binary_cross_entropy_with_logits(
            cont_logits, 1.0 - batch["terminal"].float()
        )

        metrics: dict[str, float] = {"kl": kl.item()}
        if self.decoder is not None:
            dec_in = feat if self.use_recon else feat.detach()
            recon = self.decoder(dec_in)
            recon_loss = (recon - obs).pow(2).sum((-3, -2, -1)).mean()
            if self.use_recon:
                losses["recon"] = cfg.recon_scale * recon_loss
            else:  # decoder probe: trains only the decoder, representation untouched
                losses["decoder_probe"] = recon_loss
            metrics["recon_mse"] = (recon.detach() - obs).pow(2).mean().item()

        if self.use_jepa:
            losses.update(self._jepa_losses(obs, embed, post, feat, action, metrics))

        total = sum(losses.values())
        metrics.update({f"loss/{k}": v.item() for k, v in losses.items()})
        metrics["loss/total"] = total.item()
        return total, metrics, post

    def _jepa_losses(self, obs, embed, post, feat, action, metrics):
        cfg = self.cfg
        B, T = obs.shape[:2]
        with torch.no_grad():
            target = self.target(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)

        out: dict[str, torch.Tensor] = {}
        # 0-step: the posterior (which has seen o_t) should explain the target embedding of o_t.
        if cfg.jepa_post_scale:
            out["jepa_post"] = cfg.jepa_post_scale * prediction_loss(
                self.predictor(feat), target, cfg.jepa_loss
            )

        # k-step: roll the prior open-loop from every posterior state and predict future targets.
        K = min(cfg.jepa_horizon, T - 1)
        if K > 0 and cfg.jepa_scale:
            n = T - K
            state = {k: v[:, :n].reshape(B * n, *v.shape[2:]) for k, v in post.items()}
            step_losses = []
            for j in range(1, K + 1):
                a = action[:, j : n + j].reshape(B * n, -1)
                state = self.rssm.img_step(state, a)
                pred = self.predictor(self.rssm.feat(state))
                tgt = target[:, j : n + j].reshape(B * n, -1)
                step_losses.append(prediction_loss(pred, tgt, cfg.jepa_loss))
                metrics[f"jepa/pred_k{j}"] = step_losses[-1].item()
            out["jepa_pred"] = cfg.jepa_scale * torch.stack(step_losses).mean()

        # Anti-collapse regularisation on the online projections.
        z = self.projector(embed)
        var_loss, cov_loss = vicreg_terms(z)
        if cfg.vic_var:
            out["vic_var"] = cfg.vic_var * var_loss
        if cfg.vic_cov:
            out["vic_cov"] = cfg.vic_cov * cov_loss
        with torch.no_grad():
            metrics["jepa/target_std"] = target.reshape(-1, target.shape[-1]).std(0).mean().item()
            metrics["jepa/online_std"] = z.reshape(-1, z.shape[-1]).std(0).mean().item()
        return out
