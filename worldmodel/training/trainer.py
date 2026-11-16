"""Online training loop: collect -> replay -> world-model update -> imagination actor-critic update.

Set ``ac.enabled=false`` (and optionally ``train.offline_data``) to study world-model learning
in isolation, e.g. JEPA pretraining on a fixed dataset of random-policy episodes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..agent import Agent
from ..config import Config
from ..data.replay import EpisodeBuilder, ReplayBuffer
from ..envs import VecEnv, make_env
from ..eval.probes import linear_probe, open_loop_video
from ..utils.logger import Logger
from ..utils.misc import get_device, set_seed, to_torch
from ..utils.video import save_gif


def build_env_fns(cfg: Config, seed_offset: int = 0):
    e = cfg.env
    kwargs = dict(size=e.size, max_steps=e.max_steps, action_repeat=e.action_repeat, **e.kwargs)
    return [
        (lambda i=i: make_env(e.name, seed=cfg.train.seed + seed_offset + i, **kwargs))
        for i in range(e.num_envs)
    ]


def wm_parameters(agent: Agent):
    return [p for p in agent.wm.parameters() if p.requires_grad]


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.train.seed)
        self.device = get_device(cfg.train.device)
        self.logdir = Path(cfg.train.logdir) / cfg.train.run_name
        self.logdir.mkdir(parents=True, exist_ok=True)
        (self.logdir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

        self.envs = VecEnv(build_env_fns(cfg))
        obs_shape = self.envs.observation_space.shape
        self.agent = Agent(obs_shape, self.envs.action_space, cfg).to(self.device)
        self.wm_opt = torch.optim.Adam(wm_parameters(self.agent), lr=cfg.model.lr, eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.agent.ac.actor.parameters(), lr=cfg.ac.actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.agent.ac.critic.parameters(), lr=cfg.ac.critic_lr, eps=1e-5)

        self.replay = ReplayBuffer(cfg.train.replay_capacity, cfg.train.seq_len, seed=cfg.train.seed)
        if cfg.train.offline_data:
            n = self.replay.load(cfg.train.offline_data)
            print(f"loaded {n} offline episodes ({len(self.replay)} steps) from {cfg.train.offline_data}")
        self.logger = Logger(self.logdir)
        self.rng = np.random.default_rng(cfg.train.seed)
        self.env_steps = 0
        self.grad_steps = 0

        n = self.envs.num_envs
        n_params = sum(p.numel() for p in wm_parameters(self.agent))
        print(f"device={self.device} envs={n} obs={obs_shape} action_dim={self.agent.action_dim} "
              f"objective={cfg.model.objective} wm_params={n_params / 1e6:.2f}M logdir={self.logdir}")

    # ------------------------------------------------------------------ data
    def _start_collection(self):
        n = self.envs.num_envs
        obs, infos = self.envs.reset()
        self.builders = [EpisodeBuilder() for _ in range(n)]
        zero = np.zeros(self.agent.action_dim, np.float32)
        for b, o, info in zip(self.builders, obs, infos):
            b.add(o, zero, 0.0, False, info.get("state"))
        self.obs = obs
        self.reset_mask = np.ones(n, bool)
        self.policy_state = self.agent.init_policy_state(n)
        self.ep_returns = np.zeros(n)
        self.ep_lengths = np.zeros(n, int)

    def collect(self, num_iters: int, random: bool = False) -> None:
        for _ in range(num_iters):
            if random:
                env_actions, agent_actions = self.agent.random_actions(self.envs.num_envs, self.rng)
            else:
                env_actions, agent_actions, self.policy_state = self.agent.act(
                    self.obs, self.policy_state, self.reset_mask, explore=True
                )
            obs, rewards, terms, truncs, infos = self.envs.step(env_actions)
            self.reset_mask = terms | truncs
            self.ep_returns += rewards
            self.ep_lengths += 1
            zero = np.zeros(self.agent.action_dim, np.float32)
            for i, b in enumerate(self.builders):
                info = infos[i]
                if self.reset_mask[i]:
                    final_info = info["final_info"]
                    b.add(info["final_obs"], agent_actions[i], rewards[i], terms[i], final_info.get("state"))
                    self.replay.add_episode(b.finish())
                    self.logger.add({"episode/return": self.ep_returns[i], "episode/length": self.ep_lengths[i]})
                    self.ep_returns[i], self.ep_lengths[i] = 0.0, 0
                    b.add(obs[i], zero, 0.0, False, info.get("state"))
                else:
                    b.add(obs[i], agent_actions[i], rewards[i], terms[i], info.get("state"))
            self.obs = obs
            self.env_steps += self.envs.num_envs

    # ----------------------------------------------------------------- update
    def train_step(self) -> dict[str, float]:
        cfg = self.cfg
        batch = to_torch(self.replay.sample(cfg.train.batch_size), self.device)
        loss, metrics, post = self.agent.wm.loss(batch)
        self.wm_opt.zero_grad(set_to_none=True)
        loss.backward()
        metrics["grad_norm/wm"] = torch.nn.utils.clip_grad_norm_(wm_parameters(self.agent), cfg.model.grad_clip).item()
        self.wm_opt.step()
        self.agent.wm.update_target()

        if cfg.ac.enabled:
            actor_loss, critic_loss, ac_metrics = self.agent.ac.loss(self.agent.wm, post)
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            (actor_loss + critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.agent.ac.actor.parameters(), cfg.ac.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.agent.ac.critic.parameters(), cfg.ac.grad_clip)
            self.actor_opt.step()
            self.critic_opt.step()
            self.agent.ac.update_slow_critic()
            metrics.update(ac_metrics)
        self.grad_steps += 1
        return metrics

    # ------------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        cfg = self.cfg
        self.agent.eval()
        metrics: dict[str, float] = {}
        if cfg.ac.enabled and cfg.train.eval_episodes > 0:
            env = make_env(cfg.env.name, seed=10_000 + self.env_steps, size=cfg.env.size,
                           max_steps=cfg.env.max_steps, action_repeat=cfg.env.action_repeat, **cfg.env.kwargs)
            returns = []
            for _ in range(cfg.train.eval_episodes):
                obs, _ = env.reset()
                state, reset, done, total = self.agent.init_policy_state(1), np.ones(1, bool), False, 0.0
                while not done:
                    a, _, state = self.agent.act(obs[None], state, reset, explore=False)
                    obs, r, term, trunc, _ = env.step(a[0])
                    total, done, reset = total + r, term or trunc, np.zeros(1, bool)
                returns.append(total)
            metrics["eval/return"] = float(np.mean(returns))
        seq = min(cfg.train.seq_len, 50)
        metrics.update(linear_probe(self.agent.wm, self.replay, cfg.train.probe_batches,
                                    cfg.train.batch_size, seq, k=5, device=self.device))
        self.agent.train()
        return metrics

    def save_video(self) -> None:
        self.agent.eval()
        video = open_loop_video(self.agent.wm, self.replay, self.device, seq_len=min(self.cfg.train.seq_len, 50))
        self.agent.train()
        if video is not None:
            save_gif(video, self.logdir / "videos" / f"openloop_{self.env_steps:08d}.gif")

    def save_checkpoint(self, name: str = "latest") -> Path:
        path = self.logdir / f"checkpoint_{name}.pt"
        torch.save({
            "agent": self.agent.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "env_steps": self.env_steps,
            "grad_steps": self.grad_steps,
            "config": self.cfg.to_dict(),
        }, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(ckpt["agent"])
        self.wm_opt.load_state_dict(ckpt["wm_opt"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.env_steps, self.grad_steps = ckpt["env_steps"], ckpt["grad_steps"]

    # -------------------------------------------------------------------- run
    def run(self) -> dict[str, float]:
        cfg, t = self.cfg, self.cfg.train
        n = self.envs.num_envs
        self._start_collection()
        if self.env_steps < t.prefill:
            self.collect((t.prefill - self.env_steps + n - 1) // n, random=True)
        # Gradient steps per env iteration, from the replay ratio:
        # train_ratio = (batch*seq replayed steps) / (env steps)  =>  updates per iteration below.
        updates_per_iter = t.train_ratio * n / (t.batch_size * t.seq_len)
        budget = 0.0
        next_log, next_eval = self.env_steps + t.log_every, self.env_steps + t.eval_every
        next_video, next_ckpt = self.env_steps + t.video_every, self.env_steps + t.checkpoint_every
        last_row: dict[str, float] = {}

        while self.env_steps < t.total_steps:
            act_randomly = not cfg.ac.enabled  # pure WM study: keep a random behaviour policy
            self.collect(1, random=act_randomly)
            if not self.replay.ready(t.batch_size):
                continue
            budget += updates_per_iter
            while budget >= 1.0:
                self.logger.add(self.train_step())
                budget -= 1.0
            if self.env_steps >= next_log:
                last_row = self.logger.write(self.env_steps, {"grad_steps": self.grad_steps})
                next_log += t.log_every
            if self.env_steps >= next_eval:
                last_row = self.logger.write(self.env_steps, self.evaluate(), prefix_print="EVAL ")
                next_eval += t.eval_every
            if self.env_steps >= next_video:
                self.save_video()
                next_video += t.video_every
            if self.env_steps >= next_ckpt:
                self.save_checkpoint()
                next_ckpt += t.checkpoint_every

        final = self.evaluate()
        last_row = self.logger.write(self.env_steps, final, prefix_print="FINAL ")
        self.save_video()
        self.save_checkpoint()
        self.logger.close()
        self.envs.close()
        return last_row
