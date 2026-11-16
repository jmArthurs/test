import numpy as np

from worldmodel.envs import VecEnv, make_env


def test_roomnav_shapes_and_determinism():
    a, b = make_env("roomnav3d", seed=3, size=32), make_env("roomnav3d", seed=3, size=32)
    oa, ia = a.reset()
    ob, _ = b.reset()
    assert oa.shape == (32, 32, 3) and oa.dtype == np.uint8
    assert ia["state"].shape == (6,)
    np.testing.assert_array_equal(oa, ob)
    for act in [1, 1, 3, 5, 2]:
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(act)
    np.testing.assert_array_equal(oa, ob)
    assert ra == rb


def test_actions_change_view_and_truncation():
    env = make_env("roomnav3d", seed=0, size=32, max_steps=5)
    o0, _ = env.reset()
    o1, *_ = env.step(3)  # turn
    assert np.abs(o0.astype(int) - o1.astype(int)).mean() > 1
    done = False
    for _ in range(4):
        *_, term, trunc, _ = env.step(0)
        done = term or trunc
    assert done


def test_target_reward():
    env = make_env("roomnav3d", seed=0, size=16, shaping=0.0)
    env.reset()
    env.agent_pos = env.obj_pos[0] + np.array([0.0, -0.5])
    env.agent_yaw = 0.0  # facing +z, towards the target
    _, r, *_ = env.step(1)
    assert r == 1.0


def test_continuous_and_moving_objects():
    env = make_env("roomnav3d", seed=0, size=16, continuous=True, moving_objects=True)
    env.reset()
    before = env.obj_pos[1:].copy()
    obs, *_ = env.step(np.array([0.5, 0.0, -0.2]))
    assert obs.shape == (16, 16, 3)
    assert not np.allclose(before, env.obj_pos[1:])


def test_vecenv_autoreset():
    venv = VecEnv([lambda i=i: make_env("roomnav3d", seed=i, size=16, max_steps=3) for i in range(2)])
    venv.reset()
    for _ in range(3):
        obs, rew, term, trunc, infos = venv.step(np.array([1, 1]))
    assert trunc.all() and "final_obs" in infos[0]
