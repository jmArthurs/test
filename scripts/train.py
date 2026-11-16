#!/usr/bin/env python
"""Train a world model + imagination policy.

    python scripts/train.py --config configs/jepa.yaml
    python scripts/train.py --config configs/dreamer.yaml train.run_name=dreamer_s1 train.seed=1
    python scripts/train.py --config configs/debug.yaml           # 1-minute CPU smoke run
"""

import argparse

from worldmodel.config import load_config
from worldmodel.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default="", help="path to a checkpoint_*.pt to resume from")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. model.objective=jepa")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
