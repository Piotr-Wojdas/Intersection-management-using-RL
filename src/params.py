"""Centralized parameters for PPO training and SUMO environment setup.

Edit this file to tune training behavior without touching the training loop.
"""

# --- Device selection ---
# If True, the trainer will use CUDA when PyTorch can see a GPU.
# If False, training stays on CPU even if a GPU is available.
USE_CUDA_IF_AVAILABLE = True

# Optional manual override. Set to "cpu", "cuda", or "cuda:0" if you want to
# force a specific device. Leave as None to use automatic selection.
DEVICE_OVERRIDE = None

# Optional path to the SUMO installation root.
# If left as None, `agent_ppo.py` will try to detect common Windows install paths.
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"

# --- PPO hyperparameters ---
# Learning rate used by Adam for both actor and critic.
LEARNING_RATE = 3e-4

# Number of environment steps collected before each PPO update.
ROLLOUT_STEPS = 200

# Number of optimization passes over the collected rollout.
EPOCHS = 4

# Discount factor applied to future rewards.
GAMMA = 0.99

# GAE lambda controlling the bias/variance tradeoff in advantage estimation.
GAE_LAMBDA = 0.95

# PPO clipping range for the policy ratio.
CLIP_FRAC = 0.2

# --- Environment and training configuration ---
# Keys in this dictionary are passed to `SumoEnvironment(**ENV_CONFIG)`.
# `net_file` and `route_file` are injected automatically by `agent_ppo.py`
# if they are not set here.
ENV_CONFIG = {
    # Whether to launch SUMO GUI instead of headless simulation.
    "use_gui": True,
    # Total simulated seconds in one episode.
    "num_seconds": 10000,
    # Number of simulation seconds between decision points.
    "delta_time": 5,
    # Yellow-light duration inserted when switching phases.
    "yellow_time": 3,
    # Minimum green time before a junction may change phase.
    "min_green": 5,
    # Maximum green time before forcing a switch if enforce_max_green is True.
    "max_green": 50,
    # Force a phase change when max_green is reached.
    "enforce_max_green": False,
    # Keep False for multi-agent training; True collapses to a single agent.
    "single_agent": False,
    # Default reward signal used by each traffic signal.
    "reward_fn": "diff-waiting-time",
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
    # Use a positive integer (e.g. 60) to prevent unbounded backlog; -1 disables dropping.
    "max_depart_delay": 60,
    # Time in seconds after which SUMO will teleport a stuck vehicle to the end of its edge.
    # Use a positive integer to enable gridlock teleporting, -1 to disable.
    "time_to_teleport": -1,
    # Show SUMO warnings in the console.
    "sumo_warnings": True,
    # Extra command-line arguments passed directly to SUMO.
    "additional_sumo_cmd": None,
    # Rendering mode for Gymnasium/SUMO. Use None for headless training.
    "render_mode": None,
}

# Number of outer PPO update loops.
NUM_UPDATES = 50

# --- Traffic generation defaults (used by src/City_map/generate_traffic.py) ---
# Number of vehicles to generate when creating trips/routes.
TRAFFIC_NUM_VEHICLES = 500
# Duration window (seconds) over which vehicles are distributed.
TRAFFIC_MAX_TIME = 3600
# Fraction of vehicles directed to hotspot destinations (0..1)
TRAFFIC_HOTSPOT_RATIO = 0.7

# --- Standing-vehicle penalty ---
# If a vehicle's waiting time (seconds) exceeds this threshold, it counts as "long-standing".
STANDING_WAIT_THRESHOLD = 30
# How much to penalize each long-standing vehicle (scalar multiplier).
STANDING_PENALTY_WEIGHT = 1.0
