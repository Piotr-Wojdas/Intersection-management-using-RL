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
from src.utils import pad_observation, resolve_device
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

    def log(message=""):
        print(message)
        print(message, file=log_file, flush=True)

    try:
        ts_ids = env.ts_ids
        obs_dims = {ts: env.observation_spaces(ts).shape[0] for ts in ts_ids}
        act_dims = {ts: env.action_spaces(ts).n for ts in ts_ids}

        agents: dict[str, SharedPPOAgent] = {}
        for ts in ts_ids:
            agents[ts] = SharedPPOAgent(obs_dims[ts], act_dims[ts]).to(device)

        log(f"Plik wag: {weights_file_path}")
        log(f"Plik logu: {log_file_path}")

        if os.path.exists(weights_file_path):
            checkpoint = torch.load(weights_file_path, map_location=device, weights_only=False)
            state_dict = (
                checkpoint.get("model_state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            if isinstance(state_dict, dict) and all(ts in state_dict for ts in ts_ids):
                for ts in ts_ids:
                    agents[ts].load_state_dict(state_dict[ts])
                log(f"Wczytano niezależne modele z {weights_file_path}")
            else:
                for ts in ts_ids:
                    try:
                        agents[ts].load_state_dict(state_dict)
                    except Exception:
                        pass
                log(f"Wczytano model (broadcast) z {weights_file_path}")
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
                obs_ts = pad_observation(obs_ts, obs_dims[ts])
                valid_action_dim = act_dims[ts]

                with torch.no_grad():
                    logits, _ = agents[ts]._forward(obs_ts)
                    logits = agents[ts]._sanitize_logits(logits)
                    logits = agents[ts]._mask_logits(logits, valid_action_dim)
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
    play(use_gui=True)
