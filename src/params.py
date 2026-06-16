"""Centralized parameters for PPO training and SUMO environment setup.

Edit this file to tune training behavior without touching the training loop.
"""

import os
import re

# --- Device selection ---
USE_CUDA_IF_AVAILABLE = False

# force a specific device. Leave as None to use automatic selection.
DEVICE_OVERRIDE = None

# --- Performance flags ---
# Use LIBSUMO (C++ bindings) instead of TraCI (IPC) for ~2-5x faster simulation.
# Requires libsumo installed: pip install libsumo
# Must be True before any SUMO/traci import — setup_sumo.py sets the env var.
USE_LIBSUMO = True 

# Compile PPO networks with torch.compile (PyTorch 2.0+).
# Uses 'aot_eager' backend — safe on Windows CPU, no triton required.
# Falls back silently if unavailable.
USE_TORCH_COMPILE = True
# Optional path to the SUMO installation root.
# If left as None, `src.setup_sumo` will try to detect common Windows install paths.
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"

# --- Map paths ---
BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CITY_MAP2_DIR = os.path.join(BASE_DIR, "City_map_2")
NET_FILE = os.path.join(CITY_MAP2_DIR, "city2.net.xml")

# Two demand scenarios sharing the same network. Flip USE_HARD_TRAFFIC to train
# and evaluate on the harder near-saturation variant. Keep the PPO run and the
# baselines on the SAME scenario so comparisons stay apples-to-apples.
USE_HARD_TRAFFIC = True

ROUTE_FILE_EASY = os.path.join(CITY_MAP2_DIR, "city2.rou.xml")
TRIPS_FILE_EASY = os.path.join(CITY_MAP2_DIR, "city2.trips.xml")
ROUTE_FILE_HARD = os.path.join(CITY_MAP2_DIR, "city2_hard.rou.xml")
TRIPS_FILE_HARD = os.path.join(CITY_MAP2_DIR, "city2_hard.trips.xml")

ROUTE_FILE = ROUTE_FILE_HARD if USE_HARD_TRAFFIC else ROUTE_FILE_EASY
TRIPS_FILE = TRIPS_FILE_HARD if USE_HARD_TRAFFIC else TRIPS_FILE_EASY

# --- Domain randomization ---
# A single static route file replays the same arrival sequence every episode
# (the SUMO seed only jitters micro-dynamics), so a time-aware policy can overfit
# to that one realisation. When RANDOMIZE_TRAFFIC is True, training rotates over
# a POOL of route files (different demand realisations) and evaluation uses a
# HELD-OUT file never seen in training — so the score measures generalisation.
# Generate the files with:
#   python -m src.City_map_2.generate_traffic --pool [--hard]
RANDOMIZE_TRAFFIC = True
# Number of distinct training route files in the pool.
TRAFFIC_POOL_SIZE = 8


def _scenario_base(hard: bool) -> str:
    return "city2_hard" if hard else "city2"


def traffic_pool_files(hard: bool) -> list[str]:
    """Training route-file pool for a scenario (domain randomization)."""
    base = _scenario_base(hard)
    return [
        os.path.join(CITY_MAP2_DIR, f"{base}_train_{i}.rou.xml")
        for i in range(TRAFFIC_POOL_SIZE)
    ]


def eval_route_file(hard: bool) -> str:
    """Held-out evaluation route file for a scenario (never used in training)."""
    return os.path.join(CITY_MAP2_DIR, f"{_scenario_base(hard)}_eval.rou.xml")


# Active-scenario pool + held-out file (derived from USE_HARD_TRAFFIC).
TRAFFIC_POOL_FILES = traffic_pool_files(USE_HARD_TRAFFIC)
EVAL_ROUTE_FILE = eval_route_file(USE_HARD_TRAFFIC)


def resolve_eval_route_file() -> str:
    """Route file for evaluation: the held-out file when domain randomization is
    on (and present), else the active single route file. Keeps the model eval and
    the baselines on the same demand for a fair comparison."""
    if RANDOMIZE_TRAFFIC and os.path.exists(EVAL_ROUTE_FILE):
        return EVAL_ROUTE_FILE
    return ROUTE_FILE


def reward_scale_for(hard: bool) -> float:
    """Reward multiplier per scenario (hard has ~10x larger returns)."""
    return 0.03 if hard else 0.1


def eval_seeds_for(hard: bool) -> list[int]:
    """Eval seeds per scenario (more for the bimodal, saturated hard case)."""
    return [42, 137, 271, 7, 99] if hard else [42, 137, 271]


def scenario_is_hard(scenario: str | None) -> bool:
    """Resolve a --scenario value ('easy'/'hard'/None) to a bool.

    None keeps the current USE_HARD_TRAFFIC default, so scripts behave as before
    when no flag is passed.
    """
    if scenario is None:
        return USE_HARD_TRAFFIC
    return scenario == "hard"


def apply_scenario(hard: bool) -> None:
    """Reconfigure every scenario-dependent module global for the chosen
    difficulty. Call once at startup (from a --scenario flag) before training or
    evaluation reads these values.

    NOTE: consumers must read these as ``params.X`` (qualified), not via a frozen
    ``from src.params import X``, for the switch to take effect.
    """
    global USE_HARD_TRAFFIC, ROUTE_FILE, TRIPS_FILE, REWARD_SCALE
    global TRAIN_EVAL_SEED, TRAFFIC_POOL_FILES, EVAL_ROUTE_FILE
    USE_HARD_TRAFFIC = hard
    ROUTE_FILE = ROUTE_FILE_HARD if hard else ROUTE_FILE_EASY
    TRIPS_FILE = TRIPS_FILE_HARD if hard else TRIPS_FILE_EASY
    REWARD_SCALE = reward_scale_for(hard)
    TRAIN_EVAL_SEED = eval_seeds_for(hard)
    TRAFFIC_POOL_FILES = traffic_pool_files(hard)
    EVAL_ROUTE_FILE = eval_route_file(hard)


def _highest_numbered_suffix(directory: str, prefix: str, suffix: str) -> int:
    if not os.path.isdir(directory):
        return 0

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    highest = 0
    for entry in os.listdir(directory):
        match = pattern.match(entry)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


def get_next_training_run_id() -> int:
    existing_log_runs = _highest_numbered_suffix(LOGS_DIR, "trening_", ".txt")
    existing_weight_runs = _highest_numbered_suffix(
        OUTPUTS_DIR, "ppo_models_weights_", ".pth"
    )
    return max(existing_log_runs, existing_weight_runs) + 1


def build_training_artifacts(run_id: int | None = None) -> dict[str, str | int]:
    if run_id is None:
        run_id = get_next_training_run_id()
    return {
        "run_id": run_id,
        "log_file": os.path.join(LOGS_DIR, f"trening_{run_id}.txt"),
        "weights_file": os.path.join(OUTPUTS_DIR, f"ppo_models_weights_{run_id}.pth"),
    }


def get_next_eval_run_id() -> int:
    return _highest_numbered_suffix(LOGS_DIR, "ewaluacja_", ".txt") + 1


def build_eval_log_file(run_id: int | None = None) -> str:
    if run_id is None:
        run_id = get_next_eval_run_id()
    return os.path.join(LOGS_DIR, f"ewaluacja_{run_id}.txt")


def get_next_baseline_run_id(mode: str) -> int:
    return _highest_numbered_suffix(LOGS_DIR, f"baseline_{mode}_", ".txt") + 1


def build_baseline_log_file(mode: str, run_id: int | None = None) -> str:
    if run_id is None:
        run_id = get_next_baseline_run_id(mode)
    return os.path.join(LOGS_DIR, f"baseline_{mode}_{run_id}.txt")


def _get_latest_weights_file(filename_pattern: str) -> str | None:
    if not os.path.isdir(OUTPUTS_DIR):
        return None

    pattern = re.compile(filename_pattern)
    latest_run_id = -1
    latest_path = None

    for entry in os.listdir(OUTPUTS_DIR):
        match = pattern.match(entry)
        if not match:
            continue
        run_id = int(match.group(1))
        if run_id > latest_run_id:
            latest_run_id = run_id
            latest_path = os.path.join(OUTPUTS_DIR, entry)

    return latest_path


def get_latest_best_weights_file() -> str | None:
    return _get_latest_weights_file(r"^ppo_models_weights_(\d+)_best\.pth$")


def get_latest_training_weights_file() -> str | None:
    return _get_latest_weights_file(r"^ppo_models_weights_(\d+)\.pth$")


def resolve_eval_weights_file() -> str:
    """Return the best available checkpoint path.

    Search order: latest _best.pth → latest .pth → raise so the caller
    fails loudly instead of loading a non-existent hardcoded path.
    """
    latest_best = get_latest_best_weights_file()
    if latest_best is not None:
        return latest_best

    latest_training = get_latest_training_weights_file()
    if latest_training is not None:
        return latest_training

    raise FileNotFoundError(
        f"No trained weights found in {OUTPUTS_DIR}. Run training first."
    )


def build_resume_file(run_id: int) -> str:
    """Path of the full training-state checkpoint used to resume run `run_id`."""
    return os.path.join(OUTPUTS_DIR, f"ppo_models_weights_{run_id}_resume.pth")


def get_latest_resume_run_id() -> int | None:
    """Highest run id that has a resume checkpoint in OUTPUTS_DIR, else None."""
    highest = _highest_numbered_suffix(
        OUTPUTS_DIR, "ppo_models_weights_", "_resume.pth"
    )
    return highest if highest > 0 else None


def resolve_weights_for_run(run_id: int) -> str:
    """Checkpoint path for a run id: prefer *_best.pth, fall back to *.pth."""
    best = os.path.join(OUTPUTS_DIR, f"ppo_models_weights_{run_id}_best.pth")
    if os.path.exists(best):
        return best
    final = os.path.join(OUTPUTS_DIR, f"ppo_models_weights_{run_id}.pth")
    if os.path.exists(final):
        return final
    raise FileNotFoundError(
        f"Brak checkpointu dla treningu {run_id} w {OUTPUTS_DIR} "
        "(szukano *_best.pth oraz *.pth)."
    )


# Shared PPO network architecture.
PPO_HIDDEN_DIMS = [256, 128]

# Centralized critic (MAPPO, src/agent_mappo.py) architecture. Its input is the
# concatenation of every agent's observation, so the trunk is wider than the
# per-agent actor.
CENTRAL_CRITIC_HIDDEN_DIMS = [256, 256]

# --- PPO hyperparameters ---
# Learning rate used by Adam for both actor and critic.
LEARNING_RATE = 2.5e-4

# Number of environment steps collected before each PPO update.
ROLLOUT_STEPS = 800

# Number of optimization passes over the collected rollout.
EPOCHS = 4

# Discount factor applied to future rewards.
# 0.99 gives an effective horizon of ~100 steps = 500 s, which is appropriate
# for a 4000 s episode where queue effects propagate over several hundred seconds.
GAMMA = 0.99

# GAE lambda controlling the bias/variance tradeoff in advantage estimation.
GAE_LAMBDA = 0.95

# PPO clipping range for the policy ratio (the value loss is not clipped:
# returns are unnormalized, so a clipped value loss saturates and freezes
# the critic).
CLIP_FRAC = 0.2

# PPO batch and loss controls.
PPO_MINIBATCH_SIZE = 256
PPO_ENTROPY_COEF = 0.01
# Final entropy = PPO_ENTROPY_COEF * PPO_ENTROPY_FINAL_FRAC.
# 0.1 gives 0.001 at end of training (too deterministic); 0.3 gives 0.003,
# maintaining enough exploration for J7/J1 to keep improving late in training.
PPO_ENTROPY_FINAL_FRAC = 0.3
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

# Rewards are multiplied by this factor before entering the PPO buffer, to keep
# value-function targets in a range the critic can fit quickly (logged episode
# rewards stay unscaled). The policy gradient is unaffected — advantages are
# standardized — so this only tames the critic-loss / grad-norm spikes.
# The harder scenario has ~10x larger returns, so it needs a smaller scale;
# the easy value reproduces run 8 exactly.
REWARD_SCALE = reward_scale_for(USE_HARD_TRAFFIC)

# Seed for torch/numpy/random in training (SUMO traffic stays "random").
GLOBAL_SEED = 42

# Evaluation during training.
TRAIN_EVAL_EVERY_UPDATES = 5
# Seeds averaged for a stable eval signal. At saturation the outcome is bimodal
# (clears vs cascades into gridlock), so the harder scenario needs more seeds to
# stop the "best model" selection from being luck; easy keeps run 8's three.
TRAIN_EVAL_SEED = eval_seeds_for(USE_HARD_TRAFFIC)

# How often (in updates) to persist a full resume checkpoint, so a hard crash
# loses at most this many updates. Ctrl+C always saves immediately as well.
RESUME_SAVE_EVERY_UPDATES = 5

# --- Environment and training configuration ---
# Keys in this dictionary are passed to `SumoEnvironment(**ENV_CONFIG)`.
# `net_file` and `route_file` are injected automatically by `agent_ppo.py`
# if they are not set here.
ENV_CONFIG = {
    # Whether to launch SUMO GUI instead of headless simulation.
    "use_gui": True,
    # Total simulated seconds in one episode.
    "num_seconds": 4000,
    # Number of simulation seconds between decision points.
    "delta_time": 5,
    # Yellow-light duration inserted when switching phases.
    "yellow_time": 3,
    # Minimum green time before a junction may change phase.
    "min_green": 10,
    # Maximum green time before forcing a switch if enforce_max_green is True.
    "max_green": 30,
    # Force a phase change when max_green is reached.
    "enforce_max_green": True,
    # Keep False for multi-agent training; True collapses to a single agent.
    "single_agent": False,
    # Default reward signal used by each traffic signal.
    "reward_fn": "congestion-aware",
    # Optional weights when reward_fn is a list of reward functions.
    "reward_weights": None,
    # Include system-wide metrics in info dictionaries.
    "add_system_info": True,
    # Include per-intersection metrics in info dictionaries.
    "add_per_agent_info": True,
    # SUMO seed; use "random" to randomize each run.
    "sumo_seed": "random",
    # Optional list of traffic-light IDs to control. None means all.
    "ts_ids": None,
    # Follow predefined traffic-light phases instead of agent actions.
    "fixed_ts": False,
    # Maximum time (seconds) a vehicle may wait to be inserted before SUMO drops it.
    # -1 = never drop. Dropping lets the policy hide congestion by starving the
    # entry edges: the backlog is invisible to local rewards and gets deleted.
    "max_depart_delay": -1,
    # Time in seconds after which SUMO teleports a stuck vehicle to the end of its edge.
    # Teleports erase accumulated waiting time (a positive reward spike), so a
    # short window rewards gridlock; 300 (SUMO default) only clears true deadlocks.
    "time_to_teleport": 300,
    # Show SUMO warnings in the console. Set to False to reduce log spam from SUMO.
    "sumo_warnings": False,
    # Extra command-line arguments passed directly to SUMO.
    "additional_sumo_cmd": None,
    # Rendering mode for Gymnasium/SUMO. Use None for headless training.
    "render_mode": None,
}

# Number of outer PPO update loops.
NUM_UPDATES = 300

# --- Traffic generation defaults (used by src.City_map_2.generate_traffic) ---
# Number of vehicles to generate when creating trips/routes.
# Matches the demand in the committed city2.rou.xml (1200 vehicles).
TRAFFIC_NUM_VEHICLES = 1200
# Duration window (seconds) over which vehicles are distributed.
TRAFFIC_MAX_TIME = 3000
# Fixed destination edges that should be congested more often in every epoch.
# These must exist as exit edges in the current network.
TRAFFIC_HOTSPOT_DESTINATIONS = ("E6", "E10", "E13")
# Fraction of vehicles directed to hotspot destinations (0..1)
TRAFFIC_HOTSPOT_RATIO = 0.6

# Harder scenario (generate with: python -m src.City_map_2.generate_traffic --hard).
# More vehicles + a mid-episode demand peak (profile="peak") + stronger
# directional bias create the cross-blocking where coordination beats a myopic
# controller. Tune TRAFFIC_NUM_VEHICLES_HARD if the network gridlocks/stays easy.
TRAFFIC_NUM_VEHICLES_HARD = 2000
TRAFFIC_HOTSPOT_RATIO_HARD = 0.75

# --- Standing-vehicle penalty ---
# If a vehicle's waiting time (seconds) exceeds this threshold, it counts as "long-standing".
# How much to penalize each long-standing vehicle (scalar multiplier).
# With min_green=10, delta_time=5, yellow_time=3, a vehicle in a normal red cycle can wait
# up to ~30 s before its phase returns — threshold below 30 penalises unavoidable waiting.
STANDING_WAIT_THRESHOLD = 30
STANDING_PENALTY_WEIGHT = 0.3

# Weight of the per-vehicle penalty for backlogged (not yet inserted) vehicles.
# Used by TrafficSignal._pending_vehicle_penalty (not part of congestion-aware).
PENDING_VEHICLE_PENALTY_WEIGHT = 0.1

# Penalize the total queue inside the controlled network.
QUEUE_PENALTY_WEIGHT = 0.05

# MAPPO only: weight of the GLOBAL backlog penalty added to the team reward
# (per step: -weight * number_of_pending_vehicles). Backlogged vehicles are
# invisible to the per-agent congestion reward, so without this the policy can
# inflate its reward by throttling entry and stranding cars at the gate. Only
# meaningful with a shared/team objective. Main tuning knob for the MAPPO run:
# raise it if the policy leaves backlog, lower it if it over-reacts to the
# unavoidable demand peak.
MAPPO_BACKLOG_PENALTY_WEIGHT = 0.005

# Cap (seconds) used to normalise per-lane accumulated waiting time in the
# observation. Typical lane waits are tens of seconds; a small cap keeps the
# feature responsive instead of sitting near zero.
OBS_WAIT_NORM_SECONDS = 300

# Exponential waiting penalty (optional): when True, vehicles waiting longer than
# `STANDING_WAIT_THRESHOLD` incur an exponential penalty that grows with time.
# The penalty per-vehicle is: -(STANDING_PENALTY_WEIGHT) * (exp(EXP_WAIT_PENALTY_SCALE * over) - 1) / 100
# where `over = waiting_time - STANDING_WAIT_THRESHOLD`.
EXPONENTIAL_WAIT_PENALTY = False
EXP_WAIT_PENALTY_SCALE = 0.05
