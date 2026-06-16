import argparse
import os
import time

import numpy as np
import torch

from src.agent_ppo import PPOAgent, _action_mask_tensor, build_env
from src.params import (
    apply_scenario,
    build_eval_log_file,
    resolve_eval_route_file,
    resolve_eval_weights_file,
    resolve_weights_for_run,
    scenario_is_hard,
)
from src.utils import env_reset, env_step, make_log_fn, obs_to_tensor, resolve_device


def _load_checkpoint(weights_file_path: str, device: torch.device, log):
    try:
        return torch.load(weights_file_path, map_location=device, weights_only=True)
    except Exception as exc:
        # Older checkpoints store numpy scalars (act_dims), which strict
        # weights_only loading rejects. Our own files are trusted.
        log(f"weights_only=True nie powiodło się ({exc}); ładuję bez ograniczeń.")
        return torch.load(weights_file_path, map_location=device, weights_only=False)


def play(
    use_gui: bool = False,
    sleep_seconds: float = 0.15,
    weights_path: str | None = None,
    route_file: str | None = None,
):
    device = resolve_device()
    # No explicit route → use the route the model is evaluated on for the active
    # scenario (held-out file under domain randomization).
    if route_file is None:
        route_file = resolve_eval_route_file()
    env = build_env(use_gui=use_gui, route_file=route_file)

    log_file_path = build_eval_log_file()
    weights_file_path = weights_path or resolve_eval_weights_file()
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, "w", encoding="utf-8")
    log = make_log_fn(log_file)

    try:
        ts_ids = env.ts_ids
        obs_dims = {ts: int(env.observation_spaces(ts).shape[0]) for ts in ts_ids}
        act_dims = {ts: int(env.action_spaces(ts).n) for ts in ts_ids}

        agents: dict[str, PPOAgent] = {}
        for ts in ts_ids:
            agents[ts] = PPOAgent(obs_dims[ts], act_dims[ts]).to(device)

        log(f"Weights file: {weights_file_path}")
        log(f"Log file: {log_file_path}")
        log(f"Plik tras: {route_file}")

        if os.path.exists(weights_file_path):
            checkpoint = _load_checkpoint(weights_file_path, device, log)
            state_dict = (
                checkpoint.get("model_state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            if isinstance(state_dict, dict) and all(ts in state_dict for ts in ts_ids):
                for ts in ts_ids:
                    ts_state = state_dict[ts]
                    if any(k.startswith("_orig_mod.") for k in ts_state):
                        ts_state = {k.removeprefix("_orig_mod."): v for k, v in ts_state.items()}
                    agents[ts].load_state_dict(ts_state)
                log(f"Loaded per-agent models from {weights_file_path}")
            else:
                loaded = 0
                for ts in ts_ids:
                    try:
                        agents[ts].load_state_dict(state_dict)
                        loaded += 1
                    except Exception:
                        pass
                if loaded:
                    log(
                        f"Loaded broadcast model into {loaded}/{len(ts_ids)} agents "
                        f"from {weights_file_path}"
                    )
                else:
                    log("WARNING: checkpoint did not match any agent — random policies.")
        else:
            log(f"WARNING: no weights at {weights_file_path}. Running with random policy.")

        obs_dict = env_reset(env)
        reward_sum = {ts: 0.0 for ts in ts_ids}
        wait_means: list[float] = []
        last_info: dict = {}
        steps = 0

        log("Starting evaluation on city_map_2...")

        done = False
        while not done:
            actions_dict = {
                ts: agents[ts].get_greedy_action(
                    obs_to_tensor(obs_dict[ts], obs_dims[ts], device),
                    action_mask=_action_mask_tensor(env, ts, device),
                )
                for ts in ts_ids
                if ts in obs_dict
            }

            obs_dict, rewards_dict, dones_dict, info = env_step(env, actions_dict)

            if isinstance(rewards_dict, dict):
                for ts in ts_ids:
                    reward_sum[ts] += float(rewards_dict.get(ts, 0.0))
            if isinstance(info, dict):
                last_info = info
                if "system_mean_waiting_time" in info:
                    wait_means.append(float(info["system_mean_waiting_time"]))

            steps += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            if steps % 100 == 0:
                log(
                    f"[eval] step={steps} running={info.get('system_total_running')} "
                    f"backlogged={info.get('system_total_backlogged')} "
                    f"teleported={info.get('system_total_teleported')} "
                    f"mean_wait={info.get('system_mean_waiting_time')}"
                )

            done = bool(dones_dict.get("__all__", False)) if isinstance(dones_dict, dict) else bool(dones_dict)

        log("Evaluation complete.")
        log(f"Steps taken: {steps}")
        for ts in ts_ids:
            log(f"Agent {ts} - total reward: {reward_sum[ts]:.3f}")
        if wait_means:
            log(f"Srednie czekanie (system): {float(np.mean(wait_means)):.1f}s")
        log(
            f"Dojechalo: {last_info.get('system_total_arrived', 'n/d')} | "
            f"teleporty: {last_info.get('system_total_teleported', 'n/d')} | "
            f"backlog na koncu: {last_info.get('system_total_backlogged', 'n/d')}"
        )

        env.close()
    finally:
        log_file.close()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ewaluacja wytrenowanego PPO na mapie city_map_2."
    )
    parser.add_argument(
        "--weights",
        default=None,
        metavar="RUN_ID|PATH",
        help="Numer treningu (np. 8) lub ścieżka do pliku wag. Numer rozwija się "
        "do ppo_models_weights_<N>_best.pth (lub _<N>.pth). "
        "Domyślnie: najnowszy *_best.pth.",
    )
    parser.add_argument(
        "--scenario",
        choices=["easy", "hard"],
        default=None,
        help="Trudność ruchu. Domyślnie bierze USE_HARD_TRAFFIC z params.py. "
        "Użyj 'easy' do oceny modeli uczonych na łatwym ruchu (np. run 8).",
    )
    parser.add_argument(
        "--show",
        type=int,
        choices=[0, 1],
        default=0,
        help="1 = wizualizacja w SUMO GUI; 0 (domyślnie) = headless.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Pauza między krokami (s); 0 = bez pauzy (szybka ocena).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    apply_scenario(scenario_is_hard(args.scenario))
    weights = args.weights
    if weights is not None and weights.isdigit():
        weights = resolve_weights_for_run(int(weights))
    play(
        use_gui=bool(args.show),
        sleep_seconds=args.sleep,
        weights_path=weights,
    )
