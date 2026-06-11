import os
import time

import torch

from src.agent_ppo import SharedPPOAgent
from src.params import (
    ENV_CONFIG,
    NET_FILE,
    ROUTE_FILE,
    build_eval_log_file,
    resolve_eval_weights_file,
)
from src.utils import env_reset, env_step, make_log_fn, pad_observation, resolve_device
from src.Sumo.sumo_rl import SumoEnvironment


def play(use_gui: bool | None = None, sleep_seconds: float = 0.15):
    device = resolve_device()

    env_args = dict(ENV_CONFIG)
    env_args.setdefault("net_file", NET_FILE)
    env_args.setdefault("route_file", ROUTE_FILE)
    if use_gui is not None:
        env_args["use_gui"] = use_gui
    env = SumoEnvironment(**env_args)

    log_file_path = build_eval_log_file()
    weights_file_path = resolve_eval_weights_file()
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, "w", encoding="utf-8")
    log = make_log_fn(log_file)

    try:
        ts_ids = env.ts_ids
        obs_dims = {ts: env.observation_spaces(ts).shape[0] for ts in ts_ids}
        act_dims = {ts: env.action_spaces(ts).n for ts in ts_ids}

        agents: dict[str, SharedPPOAgent] = {}
        for ts in ts_ids:
            agents[ts] = SharedPPOAgent(obs_dims[ts], act_dims[ts]).to(device)

        log(f"Weights file: {weights_file_path}")
        log(f"Log file: {log_file_path}")

        if os.path.exists(weights_file_path):
            checkpoint = torch.load(
                weights_file_path,
                map_location=device,
                weights_only=True,
            )
            state_dict = (
                checkpoint.get("model_state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            if isinstance(state_dict, dict) and all(ts in state_dict for ts in ts_ids):
                for ts in ts_ids:
                    agents[ts].load_state_dict(state_dict[ts])
                log(f"Loaded per-agent models from {weights_file_path}")
            else:
                for ts in ts_ids:
                    try:
                        agents[ts].load_state_dict(state_dict)
                    except Exception:
                        pass
                log(f"Loaded broadcast model from {weights_file_path}")
        else:
            log(f"WARNING: no weights at {weights_file_path}. Running with random policy.")

        obs_dict = env_reset(env)
        reward_sum = {ts: 0.0 for ts in ts_ids}
        steps = 0

        log("Starting evaluation on city_map_2...")

        done = False
        while not done:
            actions_dict = {
                ts: agents[ts].get_greedy_action(
                    pad_observation(
                        torch.tensor(obs_dict[ts], dtype=torch.float32, device=device),
                        obs_dims[ts],
                    ),
                    valid_action_dim=act_dims[ts],
                )
                for ts in ts_ids
                if ts in obs_dict
            }

            obs_dict, rewards_dict, dones_dict, info = env_step(env, actions_dict)

            if isinstance(rewards_dict, dict):
                for ts in ts_ids:
                    reward_sum[ts] += float(rewards_dict.get(ts, 0.0))

            steps += 1
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

        env.close()
    finally:
        log_file.close()


if __name__ == "__main__":
    play(use_gui=True)
