"""Environment registry.

``make_env("roomnav3d", ...)``          built-in numpy ray-cast 3D room (no extra deps)
``make_env("gym:Ant-v5", ...)``         any gymnasium env rendered to pixels
"""

from __future__ import annotations

from .raycast3d import RoomNav3D
from .spaces import Box, Discrete
from .vector import VecEnv

_REGISTRY = {"roomnav3d": RoomNav3D}


def register(name: str, ctor) -> None:
    _REGISTRY[name] = ctor


def make_env(name: str, seed: int | None = None, **kwargs):
    if name.startswith("gym:"):
        from .wrappers import GymnasiumPixels

        return GymnasiumPixels(name[4:], seed=seed, **kwargs)
    if name not in _REGISTRY:
        raise KeyError(f"Unknown env {name!r}. Available: {sorted(_REGISTRY)} or 'gym:<id>'")
    return _REGISTRY[name](seed=seed, **kwargs)


__all__ = ["Box", "Discrete", "RoomNav3D", "VecEnv", "make_env", "register"]
