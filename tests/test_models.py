import pytest
import torch

from worldmodel.agent import Agent
from worldmodel.config import load_config
from worldmodel.envs import Discrete, Box
from worldmodel.models.actor_critic import lambda_return
from worldmodel.models.jepa import vicreg_terms
from worldmodel.models.rssm import RSSM


def small_cfg(objective):
    return load_config("configs/debug.yaml", [f"model.objective={objective}"])


def fake_batch(B=2, T=8, size=32, A=7):
    return {
        "obs": torch.randint(0, 255, (B, T, size, size, 3), dtype=torch.uint8),
        "action": torch.nn.functional.one_hot(torch.randint(0, A, (B, T)), A).float(),
        "reward": torch.randn(B, T),
        "terminal": torch.zeros(B, T, dtype=torch.bool),
        "state": torch.randn(B, T, 6),
    }


@pytest.mark.parametrize("objective", ["dreamer", "jepa", "hybrid"])
def test_world_model_losses_and_grads(objective):
    cfg = small_cfg(objective)
    agent = Agent((32, 32, 3), Discrete(7), cfg)
    loss, metrics, post = agent.wm.loss(fake_batch())
    assert torch.isfinite(loss)
    loss.backward()
    assert agent.wm.encoder.net[0].weight.grad is not None
    assert ("loss/recon" in metrics) == (objective != "jepa")
    assert ("loss/jepa_pred" in metrics) == (objective != "dreamer")
    if objective == "jepa":
        # The decoder probe must not shape the representation.
        assert "loss/decoder_probe" in metrics
        for p in agent.wm.target.parameters():
            assert p.grad is None
    actor_loss, critic_loss, ac_metrics = agent.ac.loss(agent.wm, post)
    (actor_loss + critic_loss).backward()
    assert all(torch.isfinite(torch.tensor(v)) for v in ac_metrics.values())


def test_ema_target_moves_toward_online():
    agent = Agent((32, 32, 3), Discrete(7), small_cfg("jepa"))
    with torch.no_grad():
        for p in agent.wm.encoder.parameters():
            p.add_(1.0)
    tgt = next(agent.wm.target.parameters()).clone()
    agent.wm.update_target()
    assert not torch.allclose(tgt, next(agent.wm.target.parameters()))


def test_continuous_agent_acts():
    agent = Agent((32, 32, 3), Box(-1, 1, (3,)), small_cfg("hybrid"))
    import numpy as np

    state = agent.init_policy_state(2)
    env_a, vec, state = agent.act(np.zeros((2, 32, 32, 3), np.uint8), state, np.ones(2, bool))
    assert env_a.shape == (2, 3) and np.all(np.abs(env_a) <= 1)


def test_rssm_shapes():
    rssm = RSSM(embed_dim=10, action_dim=3, deter=16, stoch=4, classes=5, hidden=16)
    post, prior = rssm.observe(torch.randn(2, 6, 10), torch.randn(2, 6, 3))
    assert post["stoch"].shape == (2, 6, 20) and prior["logits"].shape == (2, 6, 4, 5)
    imag = rssm.imagine({k: v[:, -1] for k, v in post.items()}, torch.randn(2, 4, 3))
    assert rssm.feat(imag).shape == (2, 4, rssm.feat_dim)


def test_lambda_return_matches_discounted_sum_when_lam_is_one():
    r = torch.ones(3, 1)
    v = torch.zeros(4, 1)
    ret = lambda_return(r, v, torch.full((3, 1), 0.5), lam=1.0)
    assert torch.allclose(ret[:, 0], torch.tensor([1.75, 1.5, 1.0]))


def test_vicreg_penalises_collapse():
    var_c, _ = vicreg_terms(torch.zeros(64, 8))
    var_s, _ = vicreg_terms(torch.randn(64, 8) * 2)
    assert var_c > 0.9 and var_s < 0.1
