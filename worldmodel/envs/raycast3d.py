"""RoomNav3D: a first-person 3D navigation environment rendered by a vectorized numpy ray caster.

The scene is a closed room (checkered floor, ceiling, four distinctly coloured textured walls)
containing spheres. One sphere is the *target*: touching it yields reward and respawns it
elsewhere. The others are distractors that block movement and can optionally move around,
which gives the world model non-trivial object dynamics to predict.

Coordinates: x/z span the floor, y points up. Yaw 0 looks along +z.

The environment has zero dependencies beyond numpy, renders 64x64 in well under a millisecond
per frame on CPU, and exposes the true low-dimensional state through ``info["state"]`` so that
learned representations can be evaluated with linear probes.
"""

from __future__ import annotations

import numpy as np

from .spaces import Box, Discrete

# forward, strafe, turn
DISCRETE_ACTIONS = np.array(
    [
        [0.0, 0.0, 0.0],   # no-op
        [1.0, 0.0, 0.0],   # forward
        [-1.0, 0.0, 0.0],  # backward
        [0.0, 0.0, 1.0],   # turn left
        [0.0, 0.0, -1.0],  # turn right
        [0.0, 1.0, 0.0],   # strafe left
        [0.0, -1.0, 0.0],  # strafe right
    ],
    dtype=np.float32,
)

WALL_COLORS = np.array(
    [[0.85, 0.30, 0.25], [0.25, 0.55, 0.85], [0.30, 0.75, 0.35], [0.90, 0.80, 0.30]],
    dtype=np.float32,
)
TARGET_COLOR = np.array([1.0, 0.15, 0.6], dtype=np.float32)
DISTRACTOR_COLORS = np.array(
    [[0.2, 0.9, 0.9], [0.95, 0.55, 0.1], [0.6, 0.35, 0.9], [0.9, 0.9, 0.9], [0.4, 0.4, 0.4]],
    dtype=np.float32,
)


class RoomNav3D:
    """First-person navigation in a 3D room. Observations are ``(size, size, 3)`` uint8 images."""

    metadata = {"render_fps": 15}

    def __init__(
        self,
        size: int = 64,
        room_size: float = 4.0,
        num_distractors: int = 3,
        max_steps: int = 500,
        action_repeat: int = 1,
        fov_deg: float = 75.0,
        continuous: bool = False,
        moving_objects: bool = False,
        randomize_colors: bool = False,
        move_speed: float = 0.15,
        turn_speed: float = 0.2,
        shaping: float = 0.1,
        seed: int | None = None,
    ):
        self.size = size
        self.half = room_size / 2.0
        self.num_distractors = num_distractors
        self.max_steps = max_steps
        self.action_repeat = action_repeat
        self.continuous = continuous
        self.moving_objects = moving_objects
        self.randomize_colors = randomize_colors
        self.move_speed = move_speed
        self.turn_speed = turn_speed
        self.shaping = shaping
        self.eye_height = 0.5
        self.ceiling = 1.4
        self.agent_radius = 0.2
        self.rng = np.random.default_rng(seed)

        self.observation_space = Box(0, 255, (size, size, 3), np.uint8)
        if continuous:
            self.action_space = Box(-1.0, 1.0, (3,), np.float32)
        else:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))

        # Camera-space ray directions (x right, y up, z forward), computed once.
        t = np.tan(np.deg2rad(fov_deg) / 2.0)
        coords = (np.arange(size) + 0.5) / size * 2.0 - 1.0
        u, v = np.meshgrid(coords * t, -coords * t)
        dirs = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
        self._cam_dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        self.wall_colors = WALL_COLORS.copy()
        self._step_count = 0

    # ------------------------------------------------------------------ API
    def seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed(seed)
        self._step_count = 0
        if self.randomize_colors:
            self.wall_colors = self.rng.uniform(0.15, 0.95, (4, 3)).astype(np.float32)
        self.agent_pos = self.rng.uniform(-self.half + 0.5, self.half - 0.5, 2)
        self.agent_yaw = self.rng.uniform(-np.pi, np.pi)
        n = 1 + self.num_distractors
        self.obj_radius = np.concatenate([[0.25], self.rng.uniform(0.2, 0.4, self.num_distractors)])
        self.obj_color = np.concatenate(
            [TARGET_COLOR[None], DISTRACTOR_COLORS[np.arange(self.num_distractors) % len(DISTRACTOR_COLORS)]]
        )
        self.obj_pos = np.zeros((n, 2))
        for i in range(n):
            self.obj_pos[i] = self._free_position(self.obj_radius[i], others=range(i))
        speed = 0.05 if self.moving_objects else 0.0
        ang = self.rng.uniform(-np.pi, np.pi, n)
        self.obj_vel = speed * np.stack([np.cos(ang), np.sin(ang)], -1)
        self.obj_vel[0] = 0.0  # the target stays put
        self._prev_dist = self._target_dist()
        return self.render(), self._info()

    def step(self, action):
        a = self._action_vector(action)
        reward = 0.0
        for _ in range(self.action_repeat):
            reward += self._physics_step(a)
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        return self.render(), float(reward), False, truncated, self._info()

    # -------------------------------------------------------------- dynamics
    def _action_vector(self, action) -> np.ndarray:
        if self.continuous:
            return np.clip(np.asarray(action, np.float32).reshape(3), -1.0, 1.0)
        return DISCRETE_ACTIONS[int(np.asarray(action).reshape(()))]

    def _physics_step(self, a: np.ndarray) -> float:
        fwd, strafe, turn = a
        self.agent_yaw = (self.agent_yaw + turn * self.turn_speed + np.pi) % (2 * np.pi) - np.pi
        f = np.array([np.sin(self.agent_yaw), np.cos(self.agent_yaw)])
        left = np.array([-f[1], f[0]])
        self.agent_pos = self.agent_pos + self.move_speed * (fwd * f + strafe * left)
        lim = self.half - self.agent_radius
        self.agent_pos = np.clip(self.agent_pos, -lim, lim)

        # Distractor motion (bounce off walls).
        if self.moving_objects:
            self.obj_pos += self.obj_vel
            for i in range(1, len(self.obj_pos)):
                for d in range(2):
                    bound = self.half - self.obj_radius[i]
                    if abs(self.obj_pos[i, d]) > bound:
                        self.obj_pos[i, d] = np.sign(self.obj_pos[i, d]) * bound
                        self.obj_vel[i, d] *= -1

        # Distractors are solid: push the agent out.
        for i in range(1, len(self.obj_pos)):
            delta = self.agent_pos - self.obj_pos[i]
            dist = np.linalg.norm(delta) + 1e-8
            min_d = self.obj_radius[i] + self.agent_radius
            if dist < min_d:
                self.agent_pos = self.obj_pos[i] + delta / dist * min_d
        self.agent_pos = np.clip(self.agent_pos, -lim, lim)

        reward = 0.0
        dist = self._target_dist()
        if self.shaping:
            reward += self.shaping * (self._prev_dist - dist)
        if dist < self.obj_radius[0] + self.agent_radius:
            reward += 1.0
            self.obj_pos[0] = self._free_position(
                self.obj_radius[0], others=range(1, len(self.obj_pos)), min_agent_dist=1.0
            )
            dist = self._target_dist()
        self._prev_dist = dist
        return reward

    def _target_dist(self) -> float:
        return float(np.linalg.norm(self.agent_pos - self.obj_pos[0]))

    def _free_position(self, radius, others, min_agent_dist=0.6):
        """Rejection-sample a floor position away from the agent and the objects in ``others``."""
        lim = self.half - radius - 0.05
        others = list(others)
        for _ in range(100):
            p = self.rng.uniform(-lim, lim, 2)
            if np.linalg.norm(p - self.agent_pos) < min_agent_dist + radius:
                continue
            if all(np.linalg.norm(p - self.obj_pos[j]) > radius + self.obj_radius[j] + 0.1 for j in others):
                return p
        return p

    def _info(self) -> dict:
        rel = (self.obj_pos[0] - self.agent_pos) / self.half
        state = np.array(
            [
                self.agent_pos[0] / self.half,
                self.agent_pos[1] / self.half,
                np.sin(self.agent_yaw),
                np.cos(self.agent_yaw),
                rel[0],
                rel[1],
            ],
            dtype=np.float32,
        )
        return {"state": state}

    # ------------------------------------------------------------- rendering
    def render(self) -> np.ndarray:
        cy, sy = np.cos(self.agent_yaw), np.sin(self.agent_yaw)
        right = np.array([cy, 0.0, -sy])
        up = np.array([0.0, 1.0, 0.0])
        fwd = np.array([sy, 0.0, cy])
        d = self._cam_dirs @ np.stack([right, up, fwd])  # (N, 3) world directions
        o = np.array([self.agent_pos[0], self.eye_height, self.agent_pos[1]])
        n_rays = d.shape[0]
        inf = np.float64(1e9)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Floor / ceiling.
            t_floor = np.where(d[:, 1] < 0, -o[1] / d[:, 1], inf)
            t_ceil = np.where(d[:, 1] > 0, (self.ceiling - o[1]) / d[:, 1], inf)
            # Walls (closed box, so every ray hits one).
            tx = np.where(d[:, 0] > 0, (self.half - o[0]) / d[:, 0], (-self.half - o[0]) / d[:, 0])
            tz = np.where(d[:, 2] > 0, (self.half - o[2]) / d[:, 2], (-self.half - o[2]) / d[:, 2])
            tx = np.where(np.abs(d[:, 0]) < 1e-9, inf, tx)
            tz = np.where(np.abs(d[:, 2]) < 1e-9, inf, tz)
        t_wall = np.minimum(tx, tz)
        wall_id = np.where(tx < tz, np.where(d[:, 0] > 0, 0, 1), np.where(d[:, 2] > 0, 2, 3))

        # Spheres (centre height = radius, i.e. resting on the floor).
        centers = np.stack([self.obj_pos[:, 0], self.obj_radius, self.obj_pos[:, 1]], -1)
        oc = o[None] - centers  # (K, 3)
        b = d @ oc.T  # (N, K)
        c = (oc**2).sum(-1) - self.obj_radius**2
        disc = b**2 - c[None]
        t_sph = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), inf)
        t_sph = np.where(t_sph > 1e-4, t_sph, inf)
        sph_id = t_sph.argmin(-1)
        t_sph_min = t_sph[np.arange(n_rays), sph_id]

        t_all = np.stack([t_floor, t_ceil, t_wall, t_sph_min], -1)
        kind = t_all.argmin(-1)
        t = t_all[np.arange(n_rays), kind]
        p = o[None] + t[:, None] * d

        light = np.array([0.4, 0.8, 0.3])
        light /= np.linalg.norm(light)
        col = np.zeros((n_rays, 3), np.float32)

        m = kind == 0  # floor: checkerboard
        chk = (np.floor(p[m, 0] * 2) + np.floor(p[m, 2] * 2)) % 2
        col[m] = np.where(chk[:, None] > 0, 0.55, 0.3) * np.array([1.0, 0.95, 0.9])

        m = kind == 1  # ceiling
        col[m] = np.array([0.75, 0.78, 0.82])

        m = kind == 2  # walls: per-wall colour, vertical stripes, lambert
        wid = wall_id[m]
        along = np.where(wid < 2, p[m, 2], p[m, 0])
        stripe = 0.8 + 0.2 * (np.floor(along * 3) % 2)
        band = np.where(np.abs(p[m, 1] - 0.9) < 0.05, 0.6, 1.0)  # horizontal band for height cue
        normals = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]], np.float64)[wid]
        shade = 0.55 + 0.45 * np.abs(normals @ light)
        col[m] = self.wall_colors[wid] * (stripe * band * shade)[:, None]

        m = kind == 3  # spheres: lambert shading
        sid = sph_id[m]
        n = (p[m] - centers[sid]) / self.obj_radius[sid, None]
        diff = np.clip(n @ light, 0, 1)
        col[m] = self.obj_color[sid] * (0.3 + 0.7 * diff)[:, None]

        fog = np.exp(-0.06 * t)[:, None]
        col = col * fog + (1 - fog) * np.array([0.6, 0.6, 0.65])
        img = (np.clip(col, 0, 1) * 255).astype(np.uint8)
        return img.reshape(self.size, self.size, 3)

    def close(self) -> None:
        pass
