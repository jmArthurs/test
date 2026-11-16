#!/usr/bin/env python
"""Collect random-policy episodes to .npz files for offline world-model pretraining.

    python scripts/collect.py --out data/roomnav_random --episodes 200
    python scripts/train.py --config configs/jepa_offline.yaml train.offline_data=data/roomnav_random
"""

import argparse

import numpy as np

from worldmodel.config import load_config
from worldmodel.data.replay import EpisodeBuilder, ReplayBuffer
from worldmodel.envs import Discrete, make_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--sticky", type=float, default=0.8, help="prob. of repeating the last action (smoother motion)")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    e = cfg.env
    env = make_env(e.name, seed=cfg.train.seed, size=e.size, max_steps=e.max_steps,
                   action_repeat=e.action_repeat, **e.kwargs)
    rng = np.random.default_rng(cfg.train.seed)
    discrete = isinstance(env.action_space, Discrete) or hasattr(env.action_space, "n")
    dim = env.action_space.n if discrete else int(np.prod(env.action_space.shape))
    buf = ReplayBuffer(capacity=10**12, seq_len=1)
    for ep in range(args.episodes):
        obs, info = env.reset()
        b = EpisodeBuilder()
        b.add(obs, np.zeros(dim, np.float32), 0.0, False, info.get("state"))
        a, done = env.action_space.sample(rng), False
        while not done:
            if rng.random() > args.sticky:
                a = env.action_space.sample(rng)
            obs, r, term, trunc, info = env.step(a)
            vec = np.eye(dim, dtype=np.float32)[a] if discrete else np.asarray(a, np.float32)
            b.add(obs, vec, r, term, info.get("state"))
            done = term or trunc
        buf.add_episode(b.finish())
        print(f"episode {ep + 1}/{args.episodes} steps={len(buf)}", end="\r")
    buf.save(args.out)
    print(f"\nsaved {len(buf.episodes)} episodes ({len(buf)} steps) to {args.out}")


if __name__ == "__main__":
    main()
