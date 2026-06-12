# Traffic Signal Control with Multi-Agent PPO

Reinforcement-learning controller that manages the traffic lights of a small city
network in **SUMO** (Simulation of Urban Mobility) to minimise travel time and
queueing. Each intersection is controlled by its own **PPO** agent (Independent
PPO / IPPO); the agents learn to pick green phases that keep traffic flowing
across the whole network.

The goal: beat classical traffic control (fixed-time programs and the
max-pressure heuristic) on congestion metrics — mean waiting time, throughput,
and residual backlog.

---

## Table of contents

- [Results](#results)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration-paramspy)
- [Usage reference](#usage-reference)
- [Reading the training logs](#reading-the-training-logs)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)

---

## Results

Evaluated greedily over fixed SUMO seeds; lower reward = more congestion penalty.
All controllers run on the **same network** ([city2.net.xml](src/City_map_2/city2.net.xml),
6 signalised intersections), differing only in the traffic demand file.

### Easy scenario — 1200 vehicles

| Controller        | Reward / agent | Mean wait | Arrived   | Backlog |
|-------------------|---------------:|----------:|-----------|--------:|
| Fixed-time        |           -262 |    6.9 s  | 1200/1200 |       0 |
| Max-pressure      |        -29.8   |    0.57 s | 1200/1200 |       0 |
| **PPO** (run 8)   |        **-29** | **0.7 s** | 1200/1200 |       0 |

PPO **beats fixed-time ~9×** and **matches max-pressure** — expected, since on
light demand a myopic heuristic is already near-optimal.

### Hard scenario — 2000 vehicles, peaked demand

Saturated, near-gridlock regime where fixed-time **fails** (leaves ~330 vehicles
unable to even enter the network):

| Controller   | Reward / agent | Mean wait | Arrived   | Backlog |
|--------------|---------------:|----------:|-----------|--------:|
| Fixed-time   |          -2452 |   22.9 s  | 1669/2000 |     258 |

This is the benchmark where coordination is expected to pay off; PPO tuning on
this scenario is the current focus (see [Configuration](#configuration-paramspy)).

---

## How it works

### Environment

The simulation backend is a **vendored copy of [sumo-rl](https://github.com/LucasAlegre/sumo-rl)**
under [src/Sumo/](src/Sumo/), lightly customised for this project. SUMO advances
the microscopic traffic simulation; the RL code talks to it over TraCI. One
decision is taken every `delta_time = 5` simulated seconds; an episode is
`num_seconds = 4000` s → **800 decisions per episode**.

### Observation (per intersection)

A flat vector built in [observations.py](src/Sumo/sumo_rl/environment/observations.py):

- current green phase (one-hot)
- phase progress `[0,1]` (how close to the forced `max_green` switch)
- incoming-lane **density** and **queue** (normalised by lane capacity)
- outgoing-lane density (exposes downstream congestion → coordination)
- normalised accumulated waiting time per lane
- **upstream** neighbours' phase progress (anticipate incoming vehicle waves)
- episode progress `[0,1]`

### Action

Discrete: choose which green phase to run next. Two transitions are masked out
so the agent can never request something the environment would override:

- during `min_green` → only *hold* is allowed;
- past `max_green` → *hold* is forbidden (the agent must switch).

Masking happens in `TrafficSignal.get_action_mask()` and is applied to the policy
logits, so every buffered action matches what SUMO actually executed.

### Reward — `congestion-aware`

Per intersection, per step ([traffic_signal.py](src/Sumo/sumo_rl/environment/traffic_signal.py)):

```
reward =  Δ(waiting time)                      # progress on clearing waits
        - QUEUE_PENALTY_WEIGHT  · total_queue   # discourage standing queues
        + standing_penalty                      # penalise long-waiting vehicles
        + starvation_penalty                    # penalise growing lane imbalance
        clipped to [-20, 20]
```

### Algorithm — Independent PPO

Each of the 6 intersections has its **own** actor-critic network (shared MLP
trunk `[256, 128]` with ELU + orthogonal init, separate actor/critic heads) in
[agent_ppo.py](src/agent_ppo.py). Key implementation details:

- **GAE(λ)** advantages, clipped policy objective, **unclipped** value loss
  (returns are unnormalised, so a clipped value loss would freeze the critic).
- **Reward scaling** (`REWARD_SCALE`) keeps value targets in a range the critic
  fits quickly; advantages are standardised, so the policy gradient is unaffected.
- **Truncation bootstrap**: episodes end by time limit, so the final state's
  value is folded into the last reward instead of being treated as terminal.
- **Action masking** during action selection *and* the PPO update.
- **LR + entropy schedules** decay over training.
- **Diagnostics** every update: `explained_var`, value/policy loss, entropy,
  `approx_KL`, `clip_frac`, `grad_norm`.
- **Checkpoint / resume**: full training state (weights, Adam momentum, RNG,
  progress) is saved periodically and on `Ctrl+C`, so a run can be resumed.

---

## Repository layout

```
.
├── pyproject.toml / uv.lock          # dependencies (managed with uv)
├── README.md
└── src/
    ├── agent_ppo.py                  # ── TRAIN (entry point) + PPO agent
    ├── eval_model.py                 # ── EVALUATE a trained model (GUI/headless)
    ├── eval_baseline.py              # ── BASELINES: fixed-time & max-pressure
    ├── params.py                     # ── ALL configuration lives here
    ├── setup_sumo.py                 # SUMO_HOME bootstrap (import side-effect)
    ├── env_config.py                 # standalone env smoke test
    ├── utils.py                      # shared helpers (device, obs, logging)
    ├── City_map_2/
    │   ├── city2.net.xml             # road network — 6 signals (J0,J1,J2,J4,J7,J10)
    │   ├── city2.rou.xml             # easy traffic (1200 vehicles)
    │   ├── city2_hard.rou.xml        # hard traffic (2000 vehicles, peaked demand)
    │   └── generate_traffic.py       # ── GENERATE traffic (easy/hard)
    ├── outputs/                      # checkpoints (*.pth)
    ├── logs/                         # training / eval / baseline logs
    └── Sumo/                         # vendored sumo-rl (env, traffic_signal, observations)
```

The four `──`-marked files are the things you run.

---

## Installation

### 1. SUMO

Install SUMO from <https://eclipse.dev/sumo/>. The code looks for it via
`SUMO_HOME`, set automatically by [setup_sumo.py](src/setup_sumo.py) from the
path in [params.py](src/params.py):

```python
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
```

Edit that constant (or set the `SUMO_HOME` environment variable) to match your
install.

### 2. Python environment

Python **3.13+**. Dependencies are pinned in `uv.lock`; with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv/` with `torch`, `gymnasium`, `pettingzoo`, `sumolib`,
`traci`, `stable-baselines3`, etc. (Plain `pip install -e .` also works.)

> Run all commands **from the repository root** so that `src` is importable
> (`python -m src.<module>`). 

---

## Quick start

```bash
# 1. (optional) regenerate traffic — files are already committed
uv run -m src.City_map_2.generate_traffic          # easy
uv run -m src.City_map_2.generate_traffic --hard    # hard

# 2. train (writes the next run id automatically)
uv run -m src.agent_ppo 

# 3. evaluate the trained model, with the SUMO GUI
uv run -m src.eval_model

# 4. compare against classical controllers
uv run -m src.eval_baseline --mode fixed
uv run -m src.eval_baseline --mode max-pressure
```

---

## Configuration ([params.py](src/params.py))

Everything is tuned in one file. The most important knobs:

| Setting | Meaning |
|---|---|
| `USE_HARD_TRAFFIC` | **Scenario switch.** `True` → train/eval on `city2_hard.rou.xml`; `False` → easy. Also selects `REWARD_SCALE` and the eval seeds. |
| `NUM_UPDATES` | Number of PPO updates (outer loop). |
| `ROLLOUT_STEPS` | Env steps collected per update (800 = one full episode). |
| `LEARNING_RATE` | Adam LR (start of the decay schedule). |
| `REWARD_SCALE` | Reward multiplier into the buffer — `0.1` (easy) / `0.03` (hard). |
| `GAMMA`, `GAE_LAMBDA` | Discount and GAE bias/variance tradeoff. |
| `PPO_*` | Clip range, minibatch size, entropy/value coefficients, grad clip. |
| `TRAIN_EVAL_EVERY_UPDATES`, `TRAIN_EVAL_SEED` | Eval cadence and seeds. |
| `RESUME_SAVE_EVERY_UPDATES` | How often a resume checkpoint is written. |
| `*_PENALTY_WEIGHT`, `STANDING_WAIT_THRESHOLD` | Reward-shaping weights. |
| `SUMO_HOME`, `DEVICE_OVERRIDE`, `GLOBAL_SEED` | Environment / device / seeding. |

Scenario-dependent values (`REWARD_SCALE`, `TRAIN_EVAL_SEED`) are derived from
`USE_HARD_TRAFFIC`, so flipping that one flag reconfigures everything
consistently — and keeps the easy settings identical to the validated run 8.

---

## Usage reference

### Train

```bash
uv run -m src.agent_ppo
```

- Auto-assigns the next run id `N`; writes `logs/trening_N.txt` and
  `outputs/ppo_models_weights_N*.pth`.
- The log header prints the active **route file** so you always know which
  scenario a run used.
- The best model (by greedy eval) is saved to `..._N_best.pth` during training;
  the final model to `..._N.pth`.

**Interrupt & resume.** Press `Ctrl+C` to stop — the full training state is saved
and the resume command is printed:

```bash
uv run -m src.agent_ppo --resume 9      # resume run 9
uv run -m src.agent_ppo --resume        # resume the latest resumable run
```

Resuming continues from the next update, restoring weights, optimiser momentum,
RNG and the LR/entropy schedule position, and appends to the same log.

> Changing `REWARD_SCALE` (e.g. by flipping `USE_HARD_TRAFFIC`) invalidates a
> learned critic, so start a **fresh** run rather than resuming across that change.

### Evaluate a trained model

```bash
uv run -m src.eval_model                                  # latest model, current scenario, GUI
uv run -m src.eval_model --weights 8 --scenario easy       # run 8 on easy traffic
uv run -m src.eval_model --weights 8 --scenario easy --no-gui --sleep 0   # fast, headless
```

- `--weights` accepts a **run number** (`8` → `ppo_models_weights_8_best.pth`,
  falling back to `..._8.pth`) **or** a full path. Default: latest `*_best.pth`.
- `--scenario {easy,hard}` forces the traffic file regardless of
  `USE_HARD_TRAFFIC` — use it to evaluate a model on the scenario it was
  **trained on** (the network is identical, only demand differs).
- Results go to `logs/ewaluacja_N.txt`: per-agent reward, mean waiting time,
  arrivals, teleports, residual backlog.

### Baselines

```bash
uv run -m src.eval_baseline --mode fixed           # static signal program from the .net file
uv run -m src.eval_baseline --mode max-pressure     # greedy max-pressure heuristic
```

Reports the **same metrics** as the in-training eval (over `TRAIN_EVAL_SEED`),
so PPO and the classical controllers are directly comparable. Output:
`logs/baseline_<mode>_N.txt`. Baselines follow the active `USE_HARD_TRAFFIC`
scenario.

### Generate traffic

```bash
uv run -m src.City_map_2.generate_traffic                       # easy → city2.rou.xml
uv run -m src.City_map_2.generate_traffic --hard                # hard → city2_hard.rou.xml
uv run -m src.City_map_2.generate_traffic --num-vehicles 1600 --profile peak --seed 1
```

Samples origin→destination trips from the network's fringe edges (with a bias
toward hotspot exits), then routes them with `duarouter`. `--profile peak`
concentrates departures into a mid-episode rush; `--hard` bundles
more vehicles + peak + stronger hotspots.

---

## Reading the training logs

Each update logs a line like:

```
Diagnostyka: explained_var=0.90 | actor_loss=-0.003 | critic_loss=0.004 |
             entropia=0.31 | approx_KL=0.0008 | clip_frac=0.006 | grad_norm=0.14
```

What to watch:

| Metric | Healthy | Warning sign |
|---|---|---|
| `explained_var` | rises toward **0.8–0.95** | stuck near 0 → critic not learning |
| `critic_loss` / `grad_norm` | small, stable | spikes (10×+) → reward scale too high / high-variance scenario |
| `approx_KL` | ~0.001–0.02 | ≫0.02 → LR too high; ~0 → tiny steps, LR could rise |
| `clip_frac` | ~0.05–0.2 once warmed up | ~0 → barely updating |
| `entropia` | decays gradually | collapses to ~0 → premature determinism |

The `Ocena greedy` lines report the **real KPIs** (waiting time, arrivals,
backlog) — the numbers to compare against the baselines.

---

## Outputs

| Path | Contents |
|---|---|
| `outputs/ppo_models_weights_N_best.pth` | best checkpoint of run `N` (model only) |
| `outputs/ppo_models_weights_N.pth` | final checkpoint of run `N` |
| `outputs/ppo_models_weights_N_resume.pth` | full resume state (deleted on clean finish) |
| `logs/trening_N.txt` | training log of run `N` |
| `logs/ewaluacja_N.txt` | evaluation log |
| `logs/baseline_<mode>_N.txt` | baseline log |

---

## Troubleshooting

- **`No module named src...`** — run from the repo root, not from `src/`.
- **SUMO / TraCI not found** — fix `SUMO_HOME` in [params.py](src/params.py) or
  the environment variable; ensure `<SUMO_HOME>/tools` exists.
- **`No trained weights found`** — train first, or pass `--weights <run>` /
  a path to `eval_model`.
- **Eval looks unfair (model does badly)** — check the scenario: evaluate a
  model on the traffic it was trained on (`--scenario easy|hard`). The log header
  prints the active route file.
- **Training reward jumps wildly on the hard scenario** — that is return
  variance at saturation; lower `REWARD_SCALE` and/or add eval seeds rather than
  just adding epochs.
```
