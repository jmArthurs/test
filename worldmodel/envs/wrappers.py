"""Adapters that turn third-party environments into the pixel interface used by this project.

Interface (gymnasium-like, single environment):
    reset(seed=None) -> (obs uint8 HxWx3, info)
    step(action)     -> (obs, reward, terminated, truncated, info)
    observation_space / action_space   (see ``spaces.py``; gymnasium spaces also work)
``info["state"]`` (optional) holds a low-dim ground-truth vector used for linear probes.
"""

from __future__ import annotations

import numpy as np

from .spaces import Box, Discrete


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Area-ish resize without OpenCV: PIL if available, else nearest neighbour."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    except ImportError:
        ys = (np.arange(size) * img.shape[0] / size).astype(int)
        xs = (np.arange(size) * img.shape[1] / size).astype(int)
        return img[ys][:, xs]


class GymnasiumPixels:
    """Wrap any gymnasium env that supports ``render_mode="rgb_array"`` (e.g. MuJoCo tasks
    such as ``Ant-v5`` or ``HalfCheetah-v5``, or MiniWorld 3D mazes) as a pixel environment.

    Low-dimensional observations returned by the env are exposed as ``info["state"]``.
    """

    def __init__(self, env_id: str, size: int = 64, action_repeat: int = 2, max_steps: int = 1000,
                 seed: int | None = None, **kwargs):
        import gymnasium as gym

        self._env = gym.make(env_id, render_mode="rgb_array", **kwargs)
        self.size = size
        self.action_repeat = action_repeat
        self.max_steps = max_steps
        self._seed = seed
        self._t = 0
        self.observation_space = Box(0, 255, (size, size, 3), np.uint8)
        space = self._env.action_space
        if isinstance(space, gym.spaces.Discrete):
            self.action_space = Discrete(space.n)
        else:
            self.action_space = Box(-1.0, 1.0, space.shape, np.float32)
            self._low, self._high = space.low, space.high

    def _obs(self) -> np.ndarray:
        return resize_image(self._env.render(), self.size)

    @staticmethod
    def _state(raw) -> dict:
        if isinstance(raw, np.ndarray) and raw.ndim == 1:
            return {"state": raw.astype(np.float32)}
        return {}

    def reset(self, seed: int | None = None):
        seed = seed if seed is not None else self._seed
        self._seed = None
        raw, _ = self._env.reset(seed=seed)
        self._t = 0
        return self._obs(), self._state(raw)

    def step(self, action):
        if isinstance(self.action_space, Box):
            action = self._low + (np.clip(action, -1, 1) + 1) / 2 * (self._high - self._low)
        else:
            action = int(action)
        total = 0.0
        terminated = truncated = False
        for _ in range(self.action_repeat):
            raw, r, terminated, truncated, _ = self._env.step(action)
            total += float(r)
            if terminated or truncated:
                break
        self._t += 1
        truncated = truncated or self._t >= self.max_steps
        return self._obs(), total, bool(terminated), bool(truncated), self._state(raw)

    def close(self):
        self._env.close()
