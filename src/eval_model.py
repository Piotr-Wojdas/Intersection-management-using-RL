import os
import sys
import torch
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.Sumo.sumo_rl import SumoEnvironment
from src.agent_ppo import PPOAgent

# Try to read device / gui preferences from src.params if available
try:
    from src.params import USE_CUDA_IF_AVAILABLE, DEVICE_OVERRIDE, ENV_CONFIG
except Exception:
    USE_CUDA_IF_AVAILABLE = True
    DEVICE_OVERRIDE = None
    ENV_CONFIG = {}

if DEVICE_OVERRIDE:
    device = torch.device(DEVICE_OVERRIDE)
else:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and USE_CUDA_IF_AVAILABLE else "cpu"
    )


def play(use_gui: bool | None = None):
    if use_gui is None:
        use_gui = bool(ENV_CONFIG.get("use_gui", False))

    map_dir = os.path.join(os.path.dirname(__file__), "City_map")
    env_args = dict(ENV_CONFIG)
    env_args.setdefault("net_file", os.path.join(map_dir, "city_map.net.xml"))
    env_args.setdefault("route_file", os.path.join(map_dir, "city_map.rou.xml"))
    env = SumoEnvironment(**env_args)

    ts_ids = env.ts_ids

    # Create agents and move to device
    agents = {}
    for ts in ts_ids:
        obs_dim = env.observation_spaces(ts).shape[0]
        act_dim = env.action_spaces(ts).n

        agents[ts] = PPOAgent(obs_dim, act_dim)
        if hasattr(agents[ts], "to"):
            agents[ts].to(device)
        agents[ts].eval()

    # Load trained weights (map to device)
    model_path = os.path.join(
        os.path.dirname(__file__), "outputs", "ppo_models_weights.pth"
    )
    if os.path.exists(model_path):
        try:
            saved_weights = torch.load(model_path, map_location=device)
        except Exception:
            saved_weights = torch.load(model_path)

        # If file contains per-ts state_dicts
        if isinstance(saved_weights, dict) and all(k in saved_weights for k in ts_ids):
            for ts in ts_ids:
                try:
                    agents[ts].load_state_dict(saved_weights[ts])
                except Exception as e:
                    print(f"Nie można wczytać wag dla {ts}: {e}")
            print(f"Pomyślnie wczytano pamięć operacyjną agentów z pliku {model_path}.")
        else:
            # Assume single state_dict for all agents
            for ts in ts_ids:
                try:
                    agents[ts].load_state_dict(saved_weights)
                except Exception as e:
                    print(f"Nie można wczytać wag do agenta {ts}: {e}")
            print(
                f"Wczytano wspólne wagi z pliku {model_path} (próba dopasowania do wszystkich agentów)."
            )
    else:
        print(f"UWAGA: Nie odnaleziono wytrenowanych wag w {model_path}.")
        print("Agenci użyją losowego 'niemowlęcego' myślenia!")

    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):
        obs_dict = obs_dict[0]

    print("\nZaczynamy fizyczną jazdę!")

    # Metrics
    rewards_sum = {ts: 0.0 for ts in ts_ids}
    steps = 0

    done = False
    while not done:
        actions_dict = {}
        for ts in ts_ids:
            obs_ts = torch.tensor(obs_dict[ts], dtype=torch.float32, device=device)
            obs_in = obs_ts.unsqueeze(0) if obs_ts.dim() == 1 else obs_ts

            with torch.no_grad():
                logits = agents[ts].actor(obs_in)
                if logits.dim() > 1:
                    logits = logits.squeeze(0)
                action = int(torch.argmax(logits).item())
                actions_dict[ts] = action

        # Step environment
        res = env.step(actions_dict)
        # Unpack step results robustly for single-agent (5-tuple) and multi-agent (4-tuple)
        if len(res) == 5:
            obs_dict, rewards_dict, terminated, truncated, info = res
            dones_dict = {"__all__": bool(terminated or truncated)}
        else:
            obs_dict, rewards_dict, dones_dict, info = res

        # Accumulate rewards (handle dict or scalar)
        if isinstance(rewards_dict, dict):
            for ts in ts_ids:
                rewards_sum[ts] += float(rewards_dict.get(ts, 0.0))
        else:
            # if environment returns scalar reward, add to all agents (fallback)
            for ts in ts_ids:
                try:
                    rewards_sum[ts] += float(rewards_dict)
                except Exception:
                    pass

        steps += 1
        time.sleep(0.05)

        # Log system-level metrics occasionally to monitor backlog/teleports
        if steps % 100 == 0:
            backlogged = info.get("system_total_backlogged", None)
            teleported = info.get("system_total_teleported", None)
            running = info.get("system_total_running", None)
            mean_wait = info.get("system_mean_waiting_time", None)
            print(
                f"[eval] step={steps} running={running} backlogged={backlogged} teleported={teleported} mean_wait={mean_wait}"
            )

        # Robust done handling: prefer '__all__', otherwise check if all agents finished
        if isinstance(dones_dict, dict):
            if "__all__" in dones_dict:
                done = bool(dones_dict["__all__"])
            else:
                done = all(bool(v) for v in dones_dict.values())
        else:
            done = bool(dones_dict)

    print("Ocenianie Agenta zakończone. Trasy całkowicie obsłużone.")
    print(f"Kroków wykonano: {steps}")
    for ts in ts_ids:
        print(f"Agent {ts} — suma reward: {rewards_sum[ts]:.3f}")

    env.close()


if __name__ == "__main__":
    play()
