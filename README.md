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
- [How the network learns to control traffic](#how-the-network-learns-to-control-traffic)
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

Four models were trained and cross-evaluated across both demand scenarios.
All controllers run on the same network
([city2.net.xml](src/City_map_2/city2.net.xml), 6 signalised intersections),
with the same seeds and episode length per scenario — comparisons are direct.
Mean ± std over evaluation seeds is reported; RL models are evaluated on a
**held-out route file never seen during training** (domain randomization).

| Model | Training scenario | Algorithm |
|---|---|---|
| PPO-easy | 1200 veh, uniform demand | Independent PPO |
| MAPPO-easy | 1200 veh, uniform demand | MAPPO (centralized critic) |
| PPO-hard | 2000 veh, peaked demand | Independent PPO |
| MAPPO-hard | 2000 veh, peaked demand | MAPPO (centralized critic) |

---

### Easy scenario — 1200 vehicles (3 seeds)

| Controller | Trained on | Mean wait | Arrived | Backlog | Teleports |
|---|---|---:|---:|---:|---:|
| Fixed-time | — | 6.9 s | 1200/1200 | 0 | 0 |
| MAPPO-hard | hard | 3.1 s | 1200/1200 | 0 | 2 |
| PPO-hard | hard | 1.4 s | 1200/1200 | 0 | 0 |
| Max-pressure | — | 0.6 s | 1200/1200 | 0 | 0 |
| **PPO-easy** | **easy** | **0.4 s** | 1200/1200 | 0 | **0** |
| **MAPPO-easy** | **easy** | **0.4 s** | 1200/1200 | 0 | **0** |

All controllers clear every vehicle on easy demand; the only differentiator is
waiting time. Both easy-trained models achieve **0.4 s** — beating even
max-pressure (0.6 s). This is not noise: the result is identical across all 3
seeds with zero variance. RL trained on the matched scenario learns genuinely
optimal signal timing, while the greedy max-pressure rule still loses a fraction
of a second reacting to queues that the learned policy anticipates.

Hard-trained models transfer to easy but carry a penalty from the conservative
policy they needed to avoid gridlock: PPO-hard 1.4 s, MAPPO-hard 3.1 s with 2
teleports per run.

---

### Hard scenario — 2000 vehicles, peaked demand (5 seeds, held-out route file)

| Controller | Trained on | Mean wait | Arrived | Backlog | Teleports |
|---|---|---:|---:|---:|---:|
| Fixed-time | — | 39.0 ± 4.7 s | 1301 ± 121/2000 | 505 | 23 |
| MAPPO-easy | easy | 21.0 ± 14.2 s | 1543 ± 185/2000 | 283 | 15 |
| Max-pressure | — | 14.7 ± 5.2 s | 1876 ± 78/2000 | 31 | 18 |
| PPO-easy | easy | 7.6 ± 5.5 s | 1800 ± 108/2000 | 106 | 4 |
| MAPPO-hard | hard | 9.0 ± 0.1 s | 1895 ± 5/2000 | 47 | 10 |
| **PPO-hard** | **hard** | **4.2 ± 0.4 s** | **1897 ± 11/2000** | 68 | **2** |

The hard scenario completely reshuffles the ranking.

**Fixed-time gridlocks**: only 65 % of vehicles ever arrive (1301/2000), with
39 s average wait and 505 vehicles stranded at the network boundary.

**MAPPO-easy collapses**: 1543 arrived, 21 s wait — barely better than
fixed-time, worse than max-pressure on every metric. A model that never saw
near-saturation traffic during training has no coordination strategy for it;
the centralized critic fails catastrophically when the global state it was never
trained on appears.

**PPO-easy degrades but partially holds**: 1800 arrived, 7.6 s wait — above
max-pressure's throughput is lower (1800 vs 1876) but waiting time is lower
(7.6 s vs 14.7 s). The critical problem is the variance: ±108 arrived,
±5.5 s wait. Seed 271 gives 1594 arrived and 18.4 s wait (near fixed-time
level); seed 137 gives 1880 arrived and 4.0 s wait (near PPO-hard level). The
model has no robust congestion strategy — it stumbles into good behavior on some
demand realisations and fails on others.

**PPO-hard is the overall winner**: 1897 arrived (94.9 %), 4.2 s wait,
2.4 teleports. Throughput matches the hard-trained MAPPO but waiting time is
**2.1× lower** (4.2 s vs 9.0 s) and teleports are 4× fewer. Critically, the
variance is minimal (±11 arrived, ±0.4 s wait) — the policy is robust across
all demand seeds, including the held-out file never seen in training.

**PPO-hard beats max-pressure 3.5× on waiting time** (4.2 s vs 14.7 s) while
matching its throughput. Max-pressure is local and reactive; it cannot prevent
cross-blocking on the hotspot corridor. PPO-hard learned cross-intersection
coordination from domain randomization over 8 distinct demand realisations.

---

### Cross-scenario transfer summary

Training on the hard scenario produces models that **transfer reasonably well**
to easy (PPO-hard: 1.4 s, only 3.5× worse than the easy-specialised model).
Training on the easy scenario produces models that **fail to transfer** to hard:
MAPPO-easy catastrophically (21.0 s wait, 1543 arrived), PPO-easy unreliably
(7.6 s wait but ±5.5 s variance). The asymmetry is expected — hard training
forces the agent to learn congestion avoidance and coordination, which are
general skills. Easy training only teaches smooth flow at low load, which
provides no preparation for near-saturation regimes.

### PPO vs MAPPO

PPO consistently outperforms MAPPO in the experiments:

- **On easy**: identical throughput, but MAPPO-hard generates 2 teleports per
  run even on a 1200-vehicle scenario — a sign the centralized critic can create
  local gridlock the team reward cannot resolve.
- **On hard**: PPO-hard 2.1× lower waiting time (4.2 s vs 9.0 s), 4× fewer
  teleports. MAPPO's 135-dim global state is hard to learn from 800-step
  rollouts, and the team reward propagates noise across all 6 agents when any
  one intersection hits a congestion spike.
- MAPPO's lower end-of-episode backlog (47 vs 68) on hard reflects the global
  backlog penalty throttling network entry, but that conservatism costs waiting
  time for vehicles already inside.

---

## How it works

### Environment

The simulation backend is a **vendored copy of [sumo-rl](https://github.com/LucasAlegre/sumo-rl)**
under [src/Sumo/](src/Sumo/), lightly customised for this project. SUMO advances
the microscopic traffic simulation; the RL code talks to it over TraCI. One
decision is taken every `delta_time = 5` simulated seconds; an episode is
`num_seconds = 4000` s → **800 decisions per episode**.

### Demand & domain randomization

A SUMO route file fixes the whole demand realisation — *which* vehicle departs
*when*, from where, on which route. The SUMO seed only jitters micro-dynamics
(driver imperfection, lane tie-breaks), **not** the demand. So a single route
file replays the same arrival sequence every episode, and because the
observation includes a clock (`episode progress`), a policy can overfit to that
one timeline instead of learning general control.

To prevent this, **domain randomization** (`RANDOMIZE_TRAFFIC = True`) trains on
a **pool** of route files — one random realisation per episode — and evaluates on
a **held-out** file never seen in training, so the score measures generalisation.
The baselines are evaluated on the same held-out file, keeping the comparison
fair. Generate the files once with `generate_traffic --pool` (see
[usage](#generate-traffic)).

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
reward =  Δ(waiting time)
        - QUEUE_PENALTY_WEIGHT    · total_queued
        - STANDING_PENALTY_WEIGHT · (standing_count  +  Δlane_imbalance / 100)
        clipped to [-20, 20]
```

All four terms are subtracted (negative contributions to reward):
- `Δ(waiting time)` is the **decrease** in total accumulated wait — positive when queues clear,
  negative when they grow.
- `total_queued` — number of halted vehicles on incoming lanes right now.
- `standing_count` — vehicles whose individual waiting time exceeds `STANDING_WAIT_THRESHOLD`
  (30 s = roughly one full signal cycle); penalises lane starvation.
- `Δlane_imbalance` — step-wise increase in `max(lane_waits) − min(lane_waits)`; only applied
  when imbalance is actively growing (Δ > 0), so a stable but uneven distribution is not
  penalised, only a worsening one.

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

### Algorithm — MAPPO (centralized critic)

[agent_mappo.py](src/agent_mappo.py) keeps the per-agent **actors** (so execution
and evaluation are unchanged) but replaces the per-agent critics with **one
centralized critic** over the global state (all observations concatenated), and
trains the actors on a **team reward** (the mean of per-agent rewards). The team
reward internalises the externality that breaks IPPO: an agent that floods the
bottleneck now hurts the shared objective. This is the implemented response to
the coordination ceiling described below.

The team reward also subtracts a **global backlog penalty**
(`MAPPO_BACKLOG_PENALTY_WEIGHT · pending_vehicles`). Backlogged vehicles never
reach a controlled lane, so they are invisible to the per-agent congestion
reward — without this term the policy can inflate its reward by throttling entry
and stranding cars at the gate. A global signal like total backlog only makes
sense with a shared objective, which the centralized critic provides; the
best-model selection uses the same backlog-aware score.

---

## How the network learns to control traffic

There is no dataset of "correct" signal plans. The controller learns by **trial
and error**: each agent repeatedly tries phase choices, sees how traffic
responds, and shifts toward the choices that reduce congestion. The algorithm
doing that shifting is **PPO (Proximal Policy Optimization)**.

### The reinforcement-learning loop

At every decision point (every `delta_time` seconds) an agent:

1. **observes** its local state — queues, densities, current phase, neighbours…;
2. **acts** — picks the next green phase;
3. **receives a reward** — the `congestion-aware` signal, more negative the more
   vehicles are waiting and queueing.

The aim is to maximise the **cumulative** (discounted) reward over the whole
episode, not the immediate one — so the agent must learn that holding a phase a
bit longer now can prevent a larger queue later. The discount factor `GAMMA`
controls how far ahead it effectively plans.

### Actor and critic

PPO is an **actor–critic** method — two cooperating networks:

- the **actor** is the *policy*: it maps a state to a probability distribution
  over phases. This is what actually drives the lights.
- the **critic** is the *value function*: it estimates the expected future reward
  from a state — a learned sense of "how good is this situation".

The critic controls nothing; it exists only to **judge** the actor's decisions
and give it a reference point.

### How the policy improves — advantage + clipped updates

After collecting a batch of experience, PPO computes for each action its
**advantage**: did the outcome turn out better or worse than the critic
predicted?

- advantage **positive** → the action beat expectations → make it **more** likely;
- advantage **negative** → worse than expected → make it **less** likely.

Advantages are estimated with **GAE(λ)**, which blends short- and long-horizon
returns to trade off bias against variance.

The "proximal" part is the stability mechanism. A naive policy-gradient step can
overshoot and wreck a good policy. PPO **clips** each update so the new policy
cannot move too far from the old one in one step (`CLIP_FRAC`) — many small, safe
improvements instead of a few risky leaps. The same batch is reused for a handful
of optimisation **epochs**, then thrown away.

### Exploration → exploitation

The actor is **stochastic**: early in training it samples phases broadly, which
is how it *discovers* good behaviour nobody told it about. An **entropy** bonus
rewards keeping that randomness so the policy does not commit too early. As
training proceeds the entropy bonus and learning rate **decay on a schedule**, so
the policy gradually settles from "explore" into a decisive, nearly deterministic
controller. At evaluation it acts **greedily** — always the highest-scoring phase.

### One training cycle, repeated

Each PPO **update** is:

1. **Roll out** — run the current policy for `ROLLOUT_STEPS` steps, recording
   states, actions, rewards and the critic's value estimates.
2. **Estimate advantages** — GAE over the rollout; the critic's training targets
   are the observed returns.
3. **Optimise** — for a few epochs over minibatches, nudge the actor toward
   positive-advantage actions (clipped) and fit the critic to the returns.
4. **Repeat** for `NUM_UPDATES` updates, evaluating greedily every so often and
   keeping the best checkpoint.

A useful intuition: the **critic tends to converge first**, because the policy
can only improve as fast as the value function it is judged against. Once the
critic can separate good states from bad, the actor quickly learns phase patterns
that drain queues. The hardest, last gains are about **coordination between
intersections** — where independent agents hit a ceiling.

### Multi-agent: IPPO vs MAPPO

This project runs **one PPO agent per intersection**:

- **IPPO** ([agent_ppo.py](src/agent_ppo.py)) — each agent learns purely from its
  own local reward and treats the others as part of the environment. Simple and
  scalable, but no agent has any incentive to avoid overloading a neighbour, so a
  shared bottleneck can be starved of cooperation.
- **MAPPO** ([agent_mappo.py](src/agent_mappo.py)) — keeps the local actors
  (execution is unchanged) but adds a single critic over the **global** state and
  a **team** reward, so improving the network as a whole — not just one's own
  corner — is what gets rewarded. This is the standard fix for the coordination
  ceiling above.

---

## Repository layout

```
.
├── pyproject.toml / uv.lock          # dependencies (managed with uv)
├── README.md
└── src/
    ├── agent_ppo.py                  # ── TRAIN — Independent PPO (entry point) + PPO agent
    ├── agent_mappo.py                # ── TRAIN — MAPPO (centralized critic + team reward)
    ├── eval_model.py                 # ── EVALUATE a trained model (single episode, GUI/headless)
    ├── eval_best.py                  # ── EVALUATE robustly (mean ± spread; --baselines for comparison)
    ├── eval_baseline.py              # ── BASELINES: fixed-time & max-pressure
    ├── eval_common.py                # shared eval harness (used by eval_best & eval_baseline)
    ├── params.py                     # ── ALL configuration lives here
    ├── setup_sumo.py                 # SUMO_HOME bootstrap (import side-effect)
    ├── env_config.py                 # standalone env smoke test
    ├── utils.py                      # shared helpers (device, obs, logging)
    ├── City_map_2/
    │   ├── city2.net.xml             # road network — 6 signals (J0,J1,J2,J4,J7,J10)
    │   ├── city2.rou.xml             # easy traffic (1200 vehicles)
    │   ├── city2_hard.rou.xml        # hard traffic (2000 vehicles, peaked demand)
    │   ├── city2_hard_train_*.rou.xml # domain-randomization training pool
    │   ├── city2_hard_eval.rou.xml   # held-out evaluation demand
    │   └── generate_traffic.py       # ── GENERATE traffic (single file / --pool)
    ├── outputs/                      # checkpoints (*.pth)
    ├── logs/                         # training / eval / baseline logs
    └── Sumo/                         # vendored sumo-rl (env, traffic_signal, observations)
```

The `──`-marked files are the things you run.

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
# 1. generate the traffic pool (needed once when RANDOMIZE_TRAFFIC is on)
uv run -m src.City_map_2.generate_traffic --pool --hard   # training pool + held-out

# 2. train (writes the next run id automatically)
uv run -m src.agent_ppo            # Independent PPO
uv run -m src.agent_mappo          # MAPPO (centralized critic) — for coordination

# 3. evaluate the trained model
uv run -m src.eval_model --show 1            # one episode, watch it in the SUMO GUI
uv run -m src.eval_best                       # robust: mean ± spread over the eval seeds

# 4. compare against classical controllers
uv run -m src.eval_baseline --mode fixed
uv run -m src.eval_baseline --mode max-pressure
```

Every runnable script takes the **same two flags**: `--scenario {easy,hard}`
(difficulty; default = `USE_HARD_TRAFFIC`) and, for evaluation, `--show {0,1}`
(`1` opens the SUMO GUI; default headless).

---

## Configuration ([params.py](src/params.py))

Everything is tuned in one file. The most important knobs:

| Setting | Meaning |
|---|---|
| `USE_HARD_TRAFFIC` | **Default scenario.** `True` → hard demand, `False` → easy; also selects `REWARD_SCALE` and eval seeds. Every script can override it per-run with `--scenario {easy,hard}`. |
| `RANDOMIZE_TRAFFIC` | Train on a pool of route files (one per episode) and evaluate on a held-out file. Requires the pool — generate with `generate_traffic --pool`. |
| `TRAFFIC_POOL_SIZE` | Number of route files in the training pool. |
| `MAPPO_BACKLOG_PENALTY_WEIGHT` | MAPPO only: weight of the global backlog penalty in the team reward. |
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
uv run -m src.agent_ppo                     # default scenario (USE_HARD_TRAFFIC)
uv run -m src.agent_mappo --scenario hard   # MAPPO, force the hard scenario
```

- `--scenario {easy,hard}` overrides `USE_HARD_TRAFFIC` for this run and
  reconfigures everything (route pool, held-out eval, `REWARD_SCALE`, eval seeds).
  The pool must exist for that scenario (`generate_traffic --pool [--hard]`).
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

All evaluators share one harness ([eval_common.py](src/eval_common.py)): the same
seeds × route files and the same `mean ± std` report, so the model and the
baselines are measured under **identical conditions**.

### Evaluate a trained model

```bash
# eval_model — ONE episode (good for watching / a quick number)
uv run -m src.eval_model --weights 1 --show 1               # watch in the SUMO GUI
uv run -m src.eval_model --weights 8 --scenario easy        # run 8 on easy traffic, headless

# eval_best — ROBUST: mean ± spread over the eval seeds (defensible number)
uv run -m src.eval_best  --weights 1                        # held-out × eval seeds
uv run -m src.eval_best  --weights 1 --baselines            # model vs fixed vs max-pressure, one table
uv run -m src.eval_best  --scenario easy                    # test the model on easy traffic
uv run -m src.eval_best  --routes city2_hard_eval.rou.xml other.rou.xml   # several demands
```

- `--weights` accepts a **run number** (`1` → `ppo_models_weights_1_best.pth`,
  falling back to `..._1.pth`) **or** a full path. Default: latest `*_best.pth`.
- `--scenario {easy,hard}` picks the difficulty; the eval runs on that scenario's
  **held-out** file (or the single route file if no pool exists). Works for any
  model — e.g. test a hard-trained model on `--scenario easy`.
- `--show {0,1}` — `1` opens the SUMO GUI (default headless). `--sleep` sets the
  GUI per-step pause (eval_model only).
- `eval_best` extras: `--seeds`, `--routes` (widen the average), `--baselines`
  (also run fixed-time + max-pressure on the same seeds/routes). Output:
  `logs/ewaluacja_N.txt`.

### Baselines

```bash
uv run -m src.eval_baseline --mode fixed                    # static program from the .net file
uv run -m src.eval_baseline --mode max-pressure --scenario hard
```

Same harness and flags as `eval_best` (`--scenario`, `--seeds`, `--routes`,
`--show`) → mean ± spread over the scenario's eval seeds on the **same held-out
file**, directly comparable to the model. Output: `logs/baseline_<mode>_N.txt`.
(For everything in one command, prefer `eval_best --baselines`.)

### Generate traffic

```bash
uv run -m src.City_map_2.generate_traffic                       # easy → city2.rou.xml
uv run -m src.City_map_2.generate_traffic --hard                # hard → city2_hard.rou.xml
uv run -m src.City_map_2.generate_traffic --pool --hard          # domain-randomization pool + held-out
uv run -m src.City_map_2.generate_traffic --num-vehicles 1600 --profile peak --seed 1
```

Samples origin→destination trips from the network's fringe edges (with a bias
toward hotspot exits), then routes them with `duarouter`. `--profile peak`
concentrates departures into a mid-episode rush; `--hard` bundles
more vehicles + peak + stronger hotspots. `--pool` generates `TRAFFIC_POOL_SIZE`
training files + one held-out eval file, each from a different seed (a different
demand realisation) — required when `RANDOMIZE_TRAFFIC` is on. Switching
`USE_HARD_TRAFFIC` needs the pool regenerated for that scenario.

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
- **`RANDOMIZE_TRAFFIC=True`** — generate the pool
  with `generate_traffic --pool [--hard]`, or set `RANDOMIZE_TRAFFIC = False`.
  The pool is scenario-specific, so regenerate it after flipping `USE_HARD_TRAFFIC`.
- **Eval looks unfair (model does badly)** — check the scenario: evaluate a
  model on the traffic it was trained on (`--scenario easy|hard`). The log header
  prints the active route file.
- **Training reward jumps wildly on the hard scenario** — that is return
  variance at saturation; lower `REWARD_SCALE` and/or add eval seeds rather than
  just adding epochs.
```
