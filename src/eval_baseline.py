"""Baseline controllers for city_map_2: fixed-time and max-pressure.

Usage:
    python -m src.eval_baseline --mode fixed
    python -m src.eval_baseline --mode max-pressure

Reports the same system metrics as the in-training evaluation, so PPO results
can be compared directly against classical controllers.
"""

import argparse
import os

import numpy as np

from src.agent_ppo import build_env
from src.params import TRAIN_EVAL_SEED, build_baseline_log_file
from src.utils import env_reset, env_step, make_log_fn


def _max_pressure_actions(env) -> dict:
    """Greedy max-pressure: serve the phase with the largest (in - out) halting count."""
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
                best_score = score
                best_phase = phase
        actions[ts_id] = best_phase
    return actions


def run_episode(mode: str, seed: int) -> dict:
    fixed = mode == "fixed"
    env = build_env(use_gui=False, sumo_seed=seed, fixed_ts=fixed)
    try:
        env_reset(env)
        reward_sums = {ts: 0.0 for ts in env.ts_ids}
        wait_means: list[float] = []
        last_info: dict = {}
        done = False

        while not done:
            actions = {} if fixed else _max_pressure_actions(env)
            _obs, rewards, dones, info = env_step(env, actions)

            if isinstance(rewards, dict):
                for ts, reward in rewards.items():
                    reward_sums[ts] = reward_sums.get(ts, 0.0) + float(reward)
            if isinstance(info, dict):
                last_info = info
                if "system_mean_waiting_time" in info:
                    wait_means.append(float(info["system_mean_waiting_time"]))

            done = bool(dones.get("__all__", False)) if isinstance(dones, dict) else bool(dones)

        return {
            "reward_mean_per_agent": (
                float(np.mean(list(reward_sums.values()))) if reward_sums else 0.0
            ),
            "mean_waiting_time": float(np.mean(wait_means)) if wait_means else 0.0,
            "arrived": float(last_info.get("system_total_arrived", 0.0)),
            "teleported": float(last_info.get("system_total_teleported", 0.0)),
            "backlogged_end": float(last_info.get("system_total_backlogged", 0.0)),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixed", "max-pressure"], default="fixed")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_EVAL_SEED))
    args = parser.parse_args()

    log_file_path = build_baseline_log_file(args.mode)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log = make_log_fn(log_file)
        log(f"Baseline: {args.mode} | seeds={args.seeds}")

        results = []
        for seed in args.seeds:
            metrics = run_episode(args.mode, seed)
            results.append(metrics)
            log(
                f"seed={seed}: nagroda/agenta={metrics['reward_mean_per_agent']:.3f} | "
                f"sr. czekanie={metrics['mean_waiting_time']:.1f}s | "
                f"dojechalo={metrics['arrived']:.0f} | "
                f"teleporty={metrics['teleported']:.0f} | "
                f"backlog na koncu={metrics['backlogged_end']:.0f}"
            )

        log("--- srednia po seedach ---")
        for key in results[0]:
            log(f"{key}: {float(np.mean([m[key] for m in results])):.3f}")


if __name__ == "__main__":
    main()
