"""Representation diagnostics for world models.

* ``linear_probe``  - ridge regression from latent features to ground-truth state
  (agent position, heading, relative target position). Reports held-out R^2 for
  (a) posterior features (what the model *perceives*) and
  (b) features after k open-loop imagination steps (what the model *predicts*).
  This is the standard way to compare JEPA-style and reconstruction-based representations.
* ``open_loop_video`` - observe ``context`` frames, then imagine the rest with the true
  actions and decode (requires a decoder: dreamer/hybrid, or jepa with ``decoder_probe``).
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.misc import to_torch

STATE_NAMES = ("pos_x", "pos_z", "sin_yaw", "cos_yaw", "target_dx", "target_dz")


def ridge_r2(x_tr, y_tr, x_te, y_te, alpha: float = 1e-1) -> np.ndarray:
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    x_tr = np.concatenate([x_tr, np.ones((len(x_tr), 1))], 1)
    x_te = np.concatenate([x_te, np.ones((len(x_te), 1))], 1)
    reg = alpha * len(x_tr) * np.eye(x_tr.shape[1])
    reg[-1, -1] = 0
    w = np.linalg.solve(x_tr.T @ x_tr + reg, x_tr.T @ y_tr)
    pred = x_te @ w
    ss_res = ((y_te - pred) ** 2).sum(0)
    ss_tot = ((y_te - y_te.mean(0)) ** 2).sum(0) + 1e-8
    return 1 - ss_res / ss_tot


@torch.no_grad()
def collect_features(wm, replay, batches: int, batch_size: int, seq_len: int, k: int, device):
    post_feats, imag_feats, states, imag_states = [], [], [], []
    for _ in range(batches):
        batch = to_torch(replay.sample(batch_size, seq_len), device)
        if batch["state"].shape[-1] == 0:
            return None
        B, T = batch["obs"].shape[:2]
        embed = wm.encoder(wm.preprocess(batch["obs"]))
        post, _ = wm.rssm.observe(embed, batch["action"])
        # Skip a few warm-up steps so the filter has seen some context.
        warm = min(5, T - k - 1)
        post_feats.append(wm.rssm.feat(post)[:, warm:].reshape(-1, wm.rssm.feat_dim))
        states.append(batch["state"][:, warm:].reshape(B * (T - warm), -1))
        if k > 0:
            n = T - k
            state = {key: v[:, warm:n].reshape(B * (n - warm), *v.shape[2:]) for key, v in post.items()}
            for j in range(1, k + 1):
                state = wm.rssm.img_step(state, batch["action"][:, warm + j : n + j].reshape(B * (n - warm), -1))
            imag_feats.append(wm.rssm.feat(state))
            imag_states.append(batch["state"][:, warm + k :].reshape(B * (n - warm), -1))
    cat = lambda xs: torch.cat(xs).float().cpu().numpy() if xs else None  # noqa: E731
    return cat(post_feats), cat(states), cat(imag_feats), cat(imag_states)


def linear_probe(wm, replay, batches=8, batch_size=16, seq_len=50, k=5, device="cpu") -> dict[str, float]:
    data = collect_features(wm, replay, batches, batch_size, seq_len, k, device)
    if data is None:
        return {}
    post_x, post_y, imag_x, imag_y = data
    metrics: dict[str, float] = {}
    for tag, x, y in (("post", post_x, post_y), (f"imag_k{k}", imag_x, imag_y)):
        if x is None:
            continue
        split = int(0.8 * len(x))
        r2 = ridge_r2(x[:split], y[:split], x[split:], y[split:])
        names = STATE_NAMES if len(r2) == len(STATE_NAMES) else [f"s{i}" for i in range(len(r2))]
        for name, v in zip(names, r2):
            metrics[f"probe/{tag}/{name}"] = float(v)
        metrics[f"probe/{tag}/mean_r2"] = float(np.mean(r2))
    return metrics


@torch.no_grad()
def open_loop_video(wm, replay, device, batch_size: int = 4, seq_len: int = 50, context: int = 5):
    """Returns uint8 frames ``(T, B*H, 3*W, 3)``: one row per sample showing
    [ground truth | model (red bar = imagined) | abs error], or None without a decoder."""
    if wm.decoder is None:
        return None
    batch = to_torch(replay.sample(batch_size, seq_len), device)
    obs = wm.preprocess(batch["obs"])
    embed = wm.encoder(obs[:, :context])
    post, _ = wm.rssm.observe(embed, batch["action"][:, :context])
    last = {k: v[:, -1] for k, v in post.items()}
    prior = wm.rssm.imagine(last, batch["action"][:, context:])
    feat = torch.cat([wm.rssm.feat(post), wm.rssm.feat(prior)], 1)
    model = (wm.decoder(feat) + 0.5).clamp(0, 1)
    truth = obs + 0.5
    error = ((model - truth).abs() * 2).clamp(0, 1)
    # Mark imagined frames with a red bar on top of the model row.
    model[:, context:, :2] = torch.tensor([1.0, 0.0, 0.0], device=model.device)
    rows = torch.cat([truth, model, error], 3)  # (B, T, H, 3W, C): side by side
    video = torch.cat(list(rows), 1)  # samples stacked vertically -> (T, B*H, 3W, C)
    return (video.cpu().numpy() * 255).astype(np.uint8)
