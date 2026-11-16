"""Typed configuration with YAML files and dotted command-line overrides.

    cfg = load_config("configs/jepa.yaml", ["model.jepa_horizon=8", "train.batch_size=32"])

A YAML file may set ``base: other.yaml`` (relative path) to inherit from another file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    name: str = "roomnav3d"
    size: int = 64
    num_envs: int = 4
    max_steps: int = 500
    action_repeat: int = 1
    kwargs: dict = field(default_factory=dict)  # extra constructor args (e.g. moving_objects)


@dataclass
class ModelConfig:
    # "dreamer": pixel reconstruction | "jepa": latent prediction only | "hybrid": both
    objective: str = "hybrid"
    cnn_depth: int = 32
    deter: int = 512
    stoch: int = 32
    classes: int = 32
    hidden: int = 512
    mlp_layers: int = 2
    unimix: float = 0.01
    # Dreamer losses
    free_nats: float = 1.0
    kl_dyn: float = 0.5
    kl_rep: float = 0.1
    recon_scale: float = 1.0
    reward_scale: float = 1.0
    cont_scale: float = 1.0
    # JEPA losses
    jepa_dim: int = 256
    jepa_horizon: int = 5         # open-loop prediction steps from every posterior state
    jepa_scale: float = 1.0       # multi-step prior -> target prediction
    jepa_post_scale: float = 0.5  # posterior -> target alignment (0-step)
    jepa_loss: str = "cosine"     # cosine | l1 | mse
    vic_var: float = 1.0
    vic_cov: float = 0.04
    ema_tau: float = 0.995
    # Train a decoder on *detached* latents in jepa mode, purely for visualisation.
    decoder_probe: bool = True
    lr: float = 1e-4
    grad_clip: float = 100.0


@dataclass
class ActorCriticConfig:
    enabled: bool = True
    horizon: int = 15
    gamma: float = 0.985
    lam: float = 0.95
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    entropy: float = 3e-4
    slow_critic_tau: float = 0.02
    slow_critic_reg: float = 1.0
    hidden: int = 512
    layers: int = 2
    grad_clip: float = 100.0


@dataclass
class TrainConfig:
    run_name: str = "default"
    logdir: str = "runs"
    seed: int = 0
    device: str = "auto"
    total_steps: int = 500_000     # environment steps (summed over envs)
    prefill: int = 5_000           # random-policy steps before training
    train_ratio: int = 32          # replayed steps trained on per env step (batch*seq / ratio)
    batch_size: int = 16
    seq_len: int = 64
    replay_capacity: int = 250_000
    offline_data: str = ""         # directory of .npz episodes to preload (pure WM pretraining)
    log_every: int = 1_000
    eval_every: int = 25_000
    eval_episodes: int = 4
    probe_batches: int = 8
    video_every: int = 25_000
    checkpoint_every: int = 50_000


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _merge(dc: Any, data: dict, prefix: str = "") -> None:
    for key, value in data.items():
        if not hasattr(dc, key):
            raise KeyError(f"unknown config key '{prefix}{key}'")
        current = getattr(dc, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{prefix}{key}.")
        else:
            setattr(dc, key, value)


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop("base", None)
    if base:
        merged = _load_yaml((path.parent / base).resolve())
        _deep_update(merged, data)
        return merged
    return data


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def load_config(path: str | None = None, overrides: list[str] | tuple = ()) -> Config:
    cfg = Config()
    if path:
        _merge(cfg, _load_yaml(Path(path)))
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        nested: dict = {}
        node = nested
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
        _merge(cfg, nested)
    if cfg.model.objective not in ("dreamer", "jepa", "hybrid"):
        raise ValueError(f"model.objective must be dreamer|jepa|hybrid, got {cfg.model.objective!r}")
    return cfg
