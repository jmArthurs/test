"""Episode replay buffer with fixed-length sequence sampling and npz persistence.

Each step ``t`` of an episode stores:
    obs[t]       uint8 (H, W, C) observation o_t
    action[t]    float (A,)      action that *led to* o_t (zeros at t=0; one-hot if discrete)
    reward[t]    float           reward received on the transition into o_t
    terminal[t]  bool            o_t is terminal
    state[t]     float (S,)      optional ground-truth state (for probes)
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

KEYS = ("obs", "action", "reward", "terminal", "state")


class EpisodeBuilder:
    """Accumulates one environment's in-progress episode."""

    def __init__(self):
        self.steps: dict[str, list] = {k: [] for k in KEYS}

    def add(self, obs, action, reward, terminal, state=None) -> None:
        self.steps["obs"].append(obs)
        self.steps["action"].append(np.asarray(action, np.float32))
        self.steps["reward"].append(np.float32(reward))
        self.steps["terminal"].append(bool(terminal))
        self.steps["state"].append(np.zeros(0, np.float32) if state is None else np.asarray(state, np.float32))

    def __len__(self) -> int:
        return len(self.steps["obs"])

    def finish(self) -> dict[str, np.ndarray]:
        ep = {k: np.stack(v) for k, v in self.steps.items()}
        self.steps = {k: [] for k in KEYS}
        return ep


class ReplayBuffer:
    def __init__(self, capacity: int, seq_len: int, seed: int = 0):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes: deque[dict[str, np.ndarray]] = deque()
        self.num_steps = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_steps

    def add_episode(self, ep: dict[str, np.ndarray]) -> None:
        n = len(ep["obs"])
        if n < self.seq_len:
            return  # too short to sample a full sequence
        self.episodes.append(ep)
        self.num_steps += n
        while self.num_steps > self.capacity and len(self.episodes) > 1:
            self.num_steps -= len(self.episodes.popleft()["obs"])

    def ready(self, batch_size: int) -> bool:
        return len(self.episodes) > 0

    def sample(self, batch_size: int, seq_len: int | None = None) -> dict[str, np.ndarray]:
        T = seq_len or self.seq_len
        lengths = np.array([len(ep["obs"]) - T + 1 for ep in self.episodes], np.float64)
        ep_idx = self.rng.choice(len(self.episodes), size=batch_size, p=lengths / lengths.sum())
        out = {k: [] for k in KEYS}
        for i in ep_idx:
            ep = self.episodes[i]
            start = self.rng.integers(0, len(ep["obs"]) - T + 1)
            for k in KEYS:
                out[k].append(ep[k][start : start + T])
        return {k: np.stack(v) for k, v in out.items()}

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, ep in enumerate(self.episodes):
            np.savez_compressed(directory / f"episode_{i:06d}.npz", **ep)

    def load(self, directory: str | Path) -> int:
        files = sorted(Path(directory).glob("*.npz"))
        for f in files:
            with np.load(f) as data:
                self.add_episode({k: data[k] for k in KEYS})
        return len(files)
