"""Save videos/image grids (uint8 frames (T, H, W, 3)) as GIF; falls back to PNG strips."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_gif(frames: np.ndarray, path: str | Path, fps: int = 10, scale: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, np.uint8)
    if scale > 1:
        frames = frames.repeat(scale, 1).repeat(scale, 2)
    try:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        return path
    except ImportError:
        path = path.with_suffix(".npy")
        np.save(path, frames)
        return path
