# worldmodel — JEPA × Dreamer world models in 3D environments

A compact, hackable PyTorch base project for research on **learned world models** in
first-person 3D environments. It combines two lines of work:

| | **Dreamer** (Hafner et al., V1–V3) | **JEPA** (LeCun; I-JEPA, V-JEPA, PLDM) |
|---|---|---|
| Representation learned by | reconstructing pixels | predicting *embeddings* of future observations |
| Collapse prevention | not needed (pixel target) | EMA target encoder + stop-grad + VICReg |
| Used here for | baseline objective, imagination RL | alternative / auxiliary objective |

Both share one **RSSM** latent dynamics model (GRU + categorical latents), so you can switch
the representation objective with a single flag and compare them under identical
dynamics, data and policy learning.

```
 o_t ──► Encoder ──► e_t ──┐                         EMA target encoder
                           ▼                          (no grad)  │
 h_{t-1},z_{t-1},a_t ─► RSSM prior ─► posterior z_t ─┐           ▼
                           │                          ├─► Decoder ─► ô_t        (dreamer / hybrid)
        open-loop k steps  ▼                          ├─► reward, continue heads
                   imagined state_{t+k} ─► Predictor ─┴─► ŝ_{t+k} ≈ target(o_{t+k})  (jepa / hybrid)
                           │
                           ▼
                Actor-critic trained in imagination (λ-returns, REINFORCE)
```

## Quick start

```bash
pip install -e ".[dev,viz]"            # torch, numpy, pyyaml (+ pytest, pillow, tensorboard)
python scripts/train.py --config configs/debug.yaml        # ~1 min CPU smoke run
python -m pytest -q                                          # tests
```

Full runs (a GPU is strongly recommended for 64×64 images and the default model sizes):

```bash
python scripts/train.py --config configs/dreamer.yaml   # pixel reconstruction baseline
python scripts/train.py --config configs/jepa.yaml      # reconstruction-free latent prediction
python scripts/train.py --config configs/default.yaml   # hybrid (both)
```

Any config value can be overridden from the command line with dotted keys:

```bash
python scripts/train.py --config configs/jepa.yaml \
    model.jepa_horizon=12 model.jepa_loss=l1 train.seed=3 train.run_name=jepa_l1_s3
```

Outputs go to `runs/<run_name>/`: `metrics.jsonl`, TensorBoard events (if installed),
`videos/openloop_*.gif`, `checkpoint_latest.pt`, and `config.json`.

### Pure world-model research (no RL)

```bash
python scripts/collect.py --out data/roomnav_random --episodes 300 env.kwargs="{moving_objects: true}"
python scripts/train.py --config configs/jepa_offline.yaml train.offline_data=data/roomnav_random
python scripts/evaluate.py runs/jepa_offline/checkpoint_latest.pt --probe-k 1 5 15
```

With `ac.enabled=false` the behaviour policy stays random and only the world model trains.

## The environment: `RoomNav3D`

`worldmodel/envs/raycast3d.py` — a first-person 3D room rendered by a **vectorized numpy ray
caster** (no simulator, GPU, or OpenGL needed; ~1 ms/frame at 64×64):

* checkered floor, ceiling, four distinctly coloured striped walls (parallax + orientation cues),
  lambert shading and distance fog;
* a magenta **target sphere** (+1 reward on contact, then it respawns) and solid **distractor
  spheres**, optionally **moving** (`moving_objects: true`) to give the model object dynamics
  to predict;
* discrete (7 actions: move/strafe/turn) or continuous (`continuous: true`) control;
* `randomize_colors: true` for per-episode wall colours (generalisation studies);
* `info["state"]` exposes ground truth `(x, z, sin yaw, cos yaw, target dx, target dz)`
  used by the linear probes.

Other environments: `env.name: "gym:<id>"` wraps any gymnasium env rendered to pixels
(e.g. MuJoCo `gym:Ant-v5`, or MiniWorld 3D mazes) — see `worldmodel/envs/wrappers.py`.
Register your own with `worldmodel.envs.register(name, ctor)`; the interface is just
gymnasium-style `reset()` / `step()` returning `uint8 (H, W, 3)` images.

## Objectives (`model.objective`)

All objectives share: KL-balanced prior/posterior loss with free bits (DreamerV3), symlog
reward regression and a continue (non-terminal) classifier.

* **`dreamer`** — decoder reconstructs `o_t` from the posterior state (summed squared error).
* **`jepa`** — no pixel loss reaches the representation:
  * an **EMA target** (encoder + projector, `ema_tau`) embeds each frame;
  * from every posterior state the RSSM prior is rolled **open-loop for k = 1…`jepa_horizon`**
    steps with the real actions, and a predictor maps each imagined state to the target
    embedding of `o_{t+k}` (`jepa_loss`: `cosine` | `l1` | `mse`);
  * `jepa_post_scale` aligns the posterior itself with the target of `o_t`;
  * **VICReg** variance/covariance terms (`vic_var`, `vic_cov`) on the online projections
    guard against representational collapse — monitor `jepa/target_std`.
  * With `decoder_probe: true` a decoder is trained on **detached** latents only, so you can
    still *see* what the model imagines without the pixel loss shaping the representation.
* **`hybrid`** — both losses (default).

## Evaluation

* **Policy return** — `eval/return` (greedy policy in a fresh env).
* **Linear probes** — ridge regression from latents to the true state, held-out R²:
  `probe/post/*` (perception: posterior features) and `probe/imag_k*/*` (prediction: features
  after k imagined steps with no observations). This is the main tool for comparing
  representations learned by different objectives.
* **Open-loop video** — GIF with rows `[ground truth | model | |error|]`; after 5 context frames
  the model imagines the rest from actions alone (red bar = imagined frame).

## Code map

```
worldmodel/
  config.py              dataclass config, YAML inheritance (base:), dotted CLI overrides
  agent.py               world model + actor-critic, batched stateful policy
  envs/                  RoomNav3D ray-cast env, gymnasium pixel wrapper, VecEnv, spaces
  models/
    networks.py          MLP, conv encoder/decoder, symlog, action distributions
    rssm.py              recurrent state-space model (GRU + categorical latents)
    jepa.py              EMA target, prediction losses, VICReg terms
    world_model.py       objectives: dreamer / jepa / hybrid
    actor_critic.py      imagination actor-critic, λ-returns, return normalisation
  data/replay.py         episode replay buffer, sequence sampling, npz save/load
  training/trainer.py    collect → train WM → train AC → eval/probe/video/checkpoint
  eval/probes.py         linear probes and open-loop imagination videos
scripts/                 train.py, collect.py, evaluate.py
configs/                 default (hybrid), dreamer, jepa, jepa_offline, debug
tests/                   env, model and end-to-end smoke tests
```

## Research directions this is set up for

* Reconstruction vs. latent prediction: same RSSM, compare `probe/*`, sample efficiency, return.
* Prediction horizon and loss: `jepa_horizon`, `jepa_loss`, `jepa_post_scale`.
* Collapse dynamics: ablate `vic_var` / `vic_cov` / `ema_tau`, watch `jepa/target_std`.
* Distractors and dynamics: `moving_objects`, `num_distractors`, `randomize_colors` —
  pixel reconstruction must model every pixel, JEPA can learn to ignore irrelevant detail.
* Planning in latent space (MPC / CEM over the RSSM prior) as an alternative to the actor.
* Swapping the conv encoder for a ViT and adding masked spatial prediction (I-JEPA style).

## References

* Hafner et al. *Mastering Diverse Domains through World Models* (DreamerV3), 2023.
* Hafner et al. *Dream to Control* (Dreamer), 2020; *Learning Latent Dynamics for Planning from Pixels* (PlaNet), 2019.
* LeCun. *A Path Towards Autonomous Machine Intelligence*, 2022.
* Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA), 2023.
* Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA), 2024.
* Bardes, Ponce, LeCun. *VICReg*, 2022. Grill et al. *BYOL*, 2020.
* Sobal et al. *Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models* (PLDM), 2025.
