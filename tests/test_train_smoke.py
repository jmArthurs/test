import json

from worldmodel.config import load_config
from worldmodel.training.trainer import Trainer


def test_debug_training_run(tmp_path):
    cfg = load_config("configs/debug.yaml", [
        f"train.logdir={tmp_path}", "train.total_steps=500", "train.prefill=200",
        "train.eval_every=400", "train.video_every=400", "env.max_steps=40",
    ])
    row = Trainer(cfg).run()
    run = tmp_path / "debug"
    assert (run / "checkpoint_latest.pt").exists()
    assert list((run / "videos").glob("*.gif"))
    lines = (run / "metrics.jsonl").read_text().strip().splitlines()
    assert lines and "loss/total" in json.loads(lines[0])
    assert "eval/return" in row


def test_config_overrides_and_inheritance():
    cfg = load_config("configs/jepa.yaml", ["model.jepa_horizon=3", "env.kwargs={moving_objects: true}"])
    assert cfg.model.objective == "jepa" and cfg.model.jepa_horizon == 3
    assert cfg.env.kwargs["moving_objects"] is True
    assert cfg.train.run_name == "jepa"
