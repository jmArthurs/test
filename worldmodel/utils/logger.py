"""Console + JSONL (+ optional TensorBoard) metric logging."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


class Logger:
    def __init__(self, logdir: str | Path, use_tensorboard: bool = True):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.logdir / "metrics.jsonl", "a")
        self._buffer: dict[str, list[float]] = defaultdict(list)
        self._tb = None
        self._t0 = time.time()
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(self.logdir))
            except Exception:  # tensorboard not installed
                self._tb = None

    def add(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self._buffer[k].append(float(v))

    def write(self, step: int, extra: dict[str, float] | None = None, prefix_print: str = "") -> dict:
        row = {k: float(np.mean(v)) for k, v in self._buffer.items() if v}
        row.update(extra or {})
        self._buffer.clear()
        row["time/elapsed_min"] = (time.time() - self._t0) / 60
        self._file.write(json.dumps({"step": step, **row}) + "\n")
        self._file.flush()
        if self._tb:
            for k, v in row.items():
                self._tb.add_scalar(k, v, step)
        keys = [
            k for k in row
            if k.startswith(("loss/total", "kl", "recon", "jepa/pred_k1", "episode/return", "eval", "ac/imag_return"))
            or (k.startswith("probe") and k.endswith("mean_r2"))
        ]
        msg = " | ".join(f"{k}={row[k]:.4g}" for k in sorted(keys))
        print(f"{prefix_print}[step {step:>8d}] {msg}", flush=True)
        return row

    def close(self) -> None:
        self._file.close()
        if self._tb:
            self._tb.close()
