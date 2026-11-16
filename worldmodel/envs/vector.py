"""A tiny synchronous vectorised env with auto-reset."""

from __future__ import annotations

import numpy as np


class VecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.stack([r[0] for r in results]), [r[1] for r in results]

    def step(self, actions):
        """Returns per-env (obs, reward, terminated, truncated, info). When an episode ends,
        ``info['final_obs']``/``info['final_info']`` hold the last frame and the returned obs is
        the first frame of the next episode."""
        out_obs, rews, terms, truncs, infos = [], [], [], [], []
        for env, a in zip(self.envs, actions):
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                final_obs, final_info = obs, info
                obs, info = env.reset()
                info = dict(info, final_obs=final_obs, final_info=final_info)
            out_obs.append(obs)
            rews.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        return np.stack(out_obs), np.array(rews, np.float32), np.array(terms), np.array(truncs), infos

    def close(self):
        for env in self.envs:
            env.close()
