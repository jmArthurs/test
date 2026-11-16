#!/usr/bin/env python
"""Evaluate a checkpoint: policy return, linear probes (perception + k-step prediction),
and an open-loop imagination GIF.

    python scripts/evaluate.py runs/jepa/checkpoint_latest.pt --episodes 10 --probe-k 1 5 15
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from worldmodel.config import Config, _merge
from worldmodel.data.replay import EpisodeBuilder, ReplayBuffer
from worldmodel.envs import make_env
from worldmodel.agent import Agent
from worldmodel.eval.probes import linear_probe, open_loop_video
from worldmodel.utils.misc import get_device
from worldmodel.utils.video import save_gif


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--probe-k", type=int, nargs="+", default=[1, 5, 15])
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = Config()
    _merge(cfg, ckpt["config"])
    e = cfg.env
    env = make_env(e.name, seed=12345, size=e.size, max_steps=e.max_steps, action_repeat=e.action_repeat, **e.kwargs)
    agent = Agent(env.observation_space.shape, env.action_space, cfg).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    # Roll out the (greedy) policy and record episodes for the probes.
    replay = ReplayBuffer(10**9, seq_len=50)
    returns = []
    for _ in range(args.episodes):
        obs, info = env.reset()
        b = EpisodeBuilder()
        b.add(obs, np.zeros(agent.action_dim, np.float32), 0.0, False, info.get("state"))
        state, reset, done, total = agent.init_policy_state(1), np.ones(1, bool), False, 0.0
        while not done:
            a, vec, state = agent.act(obs[None], state, reset, explore=False)
            obs, r, term, trunc, info = env.step(a[0])
            b.add(obs, vec[0], r, term, info.get("state"))
            total, done, reset = total + r, term or trunc, np.zeros(1, bool)
        replay.add_episode(b.finish())
        returns.append(total)
    results = {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns))}
    for k in args.probe_k:
        results.update(linear_probe(agent.wm, replay, batches=8, batch_size=16, seq_len=50, k=k, device=device))
    out_dir = Path(args.checkpoint).parent
    video = open_loop_video(agent.wm, replay, device)
    if video is not None:
        results["video"] = str(save_gif(video, out_dir / "eval_openloop.gif"))
    print(json.dumps(results, indent=2))
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
