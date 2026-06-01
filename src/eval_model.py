import os
import time

import torch

from src.agent_ppo import SharedPPOAgent, pad_observation
from src.params import (
    ENV_CONFIG,
    NET_FILE,
    ROUTE_FILE,
    build_eval_log_file,
    resolve_eval_weights_file,
)
from src.Sumo.sumo_rl import SumoEnvironment


try:
    from src.params import DEVICE_OVERRIDE, USE_CUDA_IF_AVAILABLE
except Exception:
    DEVICE_OVERRIDE = None
    USE_CUDA_IF_AVAILABLE = True


def resolve_device():
    if DEVICE_OVERRIDE is not None:
        return torch.device(DEVICE_OVERRIDE)
    if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_env(use_gui: bool | None = None):
    env_args = dict(ENV_CONFIG)
    env_args.setdefault("net_file", NET_FILE)
    env_args.setdefault("route_file", ROUTE_FILE)
    if use_gui is not None:
        env_args["use_gui"] = use_gui
    return SumoEnvironment(**env_args)


def play(use_gui: bool | None = None, sleep_seconds: float = 0.15):
    device = resolve_device()
    env = build_env(use_gui=use_gui)
    log_file_path = build_eval_log_file()
    weights_file_path = resolve_eval_weights_file()
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, "w", encoding="utf-8")

    def log(message=""):
        print(message)
        print(message, file=log_file, flush=True)

    try:
        ts_ids = env.ts_ids
        ts_to_idx = {ts: idx for idx, ts in enumerate(ts_ids)}
        obs_dims = {ts: env.observation_spaces(ts).shape[0] for ts in ts_ids}
        act_dims = {ts: env.action_spaces(ts).n for ts in ts_ids}
        max_obs_dim = max(obs_dims.values())
        max_act_dim = max(act_dims.values())

        agent = SharedPPOAgent(max_obs_dim, max_act_dim, len(ts_ids)).to(device)

        log(f"Plik wag: {weights_file_path}")
        log(f"Plik logu: {log_file_path}")

        if os.path.exists(weights_file_path):
            checkpoint = torch.load(weights_file_path, map_location=device)
            state_dict = (
                checkpoint.get("model_state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            agent.load_state_dict(state_dict)
            log(f"Wczytano współdzielony model z {weights_file_path}")
        else:
            log(
                f"UWAGA: nie znaleziono wag w {weights_file_path}. Agent startuje losowo."
            )

        obs_dict = env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]

        reward_sum = {ts: 0.0 for ts in ts_ids}
        steps = 0

        log("Start oceny na mapie city_map_2...")

        done = False
        while not done:
            actions_dict = {}

            for ts in ts_ids:
                if ts not in obs_dict:
                    continue

                obs_ts = torch.tensor(obs_dict[ts], dtype=torch.float32, device=device)
                obs_ts = pad_observation(obs_ts, max_obs_dim)
                ts_idx = torch.tensor(ts_to_idx[ts], dtype=torch.long, device=device)
                valid_action_dim = act_dims[ts]

                with torch.no_grad():
                    logits, _ = agent._forward(obs_ts, ts_idx)
                    logits = agent._mask_logits(logits, valid_action_dim)
                    if logits.dim() > 1:
                        logits = logits.squeeze(0)
                    action = int(torch.argmax(logits).item())
                actions_dict[ts] = action

            res = env.step(actions_dict)
            if len(res) == 5:
                obs_dict, rewards_dict, terminated, truncated, info = res
                dones_dict = {"__all__": bool(terminated or truncated)}
            else:
                obs_dict, rewards_dict, dones_dict, info = res

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

            if isinstance(dones_dict, dict):
                done = bool(dones_dict.get("__all__", False))
            else:
                done = bool(dones_dict)

        log("Ocenianie zakończone.")
        log(f"Kroków wykonano: {steps}")
        for ts in ts_ids:
            log(f"Agent {ts} - suma reward: {reward_sum[ts]:.3f}")

        env.close()
    finally:
        log_file.close()


if __name__ == "__main__":
    play(use_gui=bool(ENV_CONFIG.get("use_gui", False)))
