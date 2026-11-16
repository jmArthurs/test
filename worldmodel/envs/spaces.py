"""Minimal gym-style spaces so the core project does not depend on gymnasium."""

from __future__ import annotations

import numpy as np


class Discrete:
    def __init__(self, n: int):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.int64

    def sample(self, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        return int(rng.integers(self.n))

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def __repr__(self) -> str:
        return f"Box(shape={self.shape}, dtype={np.dtype(self.dtype).name})"
