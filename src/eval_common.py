"""Shared evaluation harness.

Runs a controller (trained model, fixed-time, or max-pressure) over the same
seeds x route files and reports each run plus mean ± spread of the KPIs. Used by
both eval_best (model) and eval_baseline (classical controllers), so the model
and the baselines are measured under *identical* conditions and are directly
comparable.

A "controller" is just an `action_fn(obs, env) -> actions_dict`:
- model        → greedy_action_fn(...)
- fixed-time   → fixed_action_fn   (empty dict; the env follows the .net program)
- max-pressure → max_pressure_action_fn
"""

import os

import numpy as np
import torch

from src.agent_ppo import PPOAgent, _action_mask_tensor, build_env
from src.params import CITY_MAP2_DIR, resolve_eval_route_file
from src.utils import env_reset, env_step, obs_to_tensor

_KPIS = ("arrived", "mean_waiting_time", "backlogged_end", "teleported")


def resolve_routes(routes_arg) -> list[str]:
    """Resolve --routes (names or paths) to absolute files; default = held-out."""
    if not routes_arg:
        return [resolve_eval_route_file()]
    return [
        r if (os.path.isabs(r) or os.path.exists(r)) else os.path.join(CITY_MAP2_DIR, r)
        for r in routes_arg
    ]


# --- Controllers -----------------------------------------------------------

def fixed_action_fn(obs, env) -> dict:
    """Empty actions → SUMO runs the fixed-time program from the .net file."""
    return {}


def max_pressure_action_fn(obs, env) -> dict:
    """Greedy max-pressure: serve the phase with the largest (in - out) queue."""
    actions = {}
    for ts_id in env.ts_ids:
        ts = env.traffic_signals[ts_id]
        halting = ts.sumo.lane.getLastStepHaltingNumber
        best_phase, best_score = 0, -float("inf")
        for phase in range(ts.num_green_phases):
            score = sum(halting(lane) for lane in ts.phase_served_lanes[phase]) - sum(
                halting(lane) for lane in ts.phase_out_lanes[phase]
            )
            if score > best_score:
                best_score, best_phase = score, phase
        actions[ts_id] = best_phase
    return actions


def load_agents(weights_path, ts_ids, obs_dims, act_dims, device):
    """Build per-agent actors and load a checkpoint (IPPO or MAPPO best)."""
    agents = {ts: PPOAgent(obs_dims[ts], act_dims[ts]).to(device) for ts in ts_ids}
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not (isinstance(state, dict) and all(ts in state for ts in ts_ids)):
        raise ValueError(f"Checkpoint {weights_path} nie pasuje do agentów {ts_ids}.")
    for ts in ts_ids:
        agents[ts].load_state_dict(state[ts])
    return agents


def greedy_action_fn(agents, ts_ids, obs_dims, device):
    """Controller that takes the argmax (deployment) action for each signal."""

    def fn(obs, env):
        return {
            ts: agents[ts].get_greedy_action(
                obs_to_tensor(obs[ts], obs_dims[ts], device),
                action_mask=_action_mask_tensor(env, ts, device),
            )
            for ts in ts_ids
            if ts in obs
        }

    return fn


def network_dims(route_file):
    """Probe a (cheap) env to read ts ids and obs/act dims for the network."""
    env = build_env(use_gui=False, route_file=route_file)
    try:
        ts_ids = env.ts_ids
        obs_dims = {ts: int(env.observation_spaces(ts).shape[0]) for ts in ts_ids}
        act_dims = {ts: int(env.action_spaces(ts).n) for ts in ts_ids}
    finally:
        env.close()
    return ts_ids, obs_dims, act_dims


# --- Evaluation ------------------------------------------------------------

def _run_episode(env, action_fn) -> dict:
    obs = env_reset(env)
    wait_means, last_info = [], {}
    done = False
    while not done:
        actions = action_fn(obs, env)
        obs, _rewards, dones, info = env_step(env, actions)
        if isinstance(info, dict):
            last_info = info
            if "system_mean_waiting_time" in info:
                wait_means.append(float(info["system_mean_waiting_time"]))
        done = bool(dones.get("__all__", False)) if isinstance(dones, dict) else bool(dones)
    return {
        "arrived": float(last_info.get("system_total_arrived", 0.0)),
        "mean_waiting_time": float(np.mean(wait_means)) if wait_means else 0.0,
        "backlogged_end": float(last_info.get("system_total_backlogged", 0.0)),
        "teleported": float(last_info.get("system_total_teleported", 0.0)),
    }


def evaluate(action_fn, routes, seeds, use_gui=False, fixed_ts=False, log=None):
    """Run a controller over every (route, seed); return per-run KPI dicts."""
    results = []
    for route in routes:
        for seed in seeds:
            env = build_env(
                use_gui=use_gui, sumo_seed=seed, route_file=route, fixed_ts=fixed_ts
            )
            try:
                metrics = _run_episode(env, action_fn)
            finally:
                env.close()
            metrics["route"], metrics["seed"] = os.path.basename(route), seed
            results.append(metrics)
            if log:
                log(
                    f"  {metrics['route']} seed={seed}: "
                    f"dojechalo={metrics['arrived']:.0f} | "
                    f"sr. czekanie={metrics['mean_waiting_time']:.1f}s | "
                    f"backlog={metrics['backlogged_end']:.0f} | "
                    f"teleporty={metrics['teleported']:.0f}"
                )
    return results


def summarize(results, log, label="") -> dict:
    """Print mean ± std (min, max) of each KPI; return the means."""
    tag = f" — {label}" if label else ""
    log(f"\n=== Srednia ± odchylenie{tag} ({len(results)} przejazdow) ===")
    means = {}
    for key in _KPIS:
        vals = np.array([r[key] for r in results], dtype=float)
        means[key] = float(vals.mean())
        log(
            f"{key}: {vals.mean():.1f} ± {vals.std():.1f} "
            f"(min {vals.min():.1f}, max {vals.max():.1f})"
        )
    return means
