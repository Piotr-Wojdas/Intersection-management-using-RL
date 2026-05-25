import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from src.params import (
    LEARNING_RATE,
    ROLLOUT_STEPS,
    EPOCHS,
    GAMMA,
    GAE_LAMBDA,
    CLIP_FRAC,
    DEVICE_OVERRIDE,
    USE_CUDA_IF_AVAILABLE,
    NUM_UPDATES,
    ENV_CONFIG,
)

from src.Sumo.sumo_rl import SumoEnvironment


# ========================================== #
# 1. Definicja sieci neuronowej (Od Zera)    #
# Zbudujemy model Actor-Critic, który będzie #
# sterował jednym skrzyżowaniem              #
# ========================================== #
class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PPOAgent, self).__init__()

        # Sieć policy (Aktor)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

        # Sieć wartości (Krytyk)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(obs).squeeze(-1),
        )


def resolve_device():
    if DEVICE_OVERRIDE is not None:
        return torch.device(DEVICE_OVERRIDE)
    if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Główna funkcja treningowa
def train():
    device = resolve_device()
    print(f"Używam urządzenia: {device}")

    # Environment parameters come from ENV_CONFIG in params.py
    map_dir = os.path.join(os.path.dirname(__file__), "City_map")
    env_args = dict(ENV_CONFIG)
    env_args.setdefault("net_file", os.path.join(map_dir, "city_map.net.xml"))
    env_args.setdefault("route_file", os.path.join(map_dir, "city_map.rou.xml"))
    env = SumoEnvironment(**env_args)

    ts_ids = env.ts_ids

    # Tworzymy osobnego agenta dla każdego skrzyżowania w mieście
    agents = {}
    optimizers = {}
    for ts in ts_ids:
        obs_dim = env.observation_spaces(ts).shape[0]
        act_dim = env.action_spaces(ts).n
        agents[ts] = PPOAgent(obs_dim, act_dim).to(device)
        optimizers[ts] = optim.Adam(agents[ts].parameters(), lr=LEARNING_RATE)

    print(f"Stworzono Agentów od zera (PyTorch) dla {len(ts_ids)} skrzyżowań.")

    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):
        obs_dict = obs_dict[0]

    for update in range(1, NUM_UPDATES + 1):
        print(f"\nUruchamiam epokę uczenia #{update}")

        # Rejestry pamięci na to czego agenci się nauczyli
        memories = {
            ts: {
                "obs": [],
                "actions": [],
                "logprobs": [],
                "rewards": [],
                "values": [],
                "dones": [],
            }
            for ts in ts_ids
        }

        for step in range(ROLLOUT_STEPS):
            actions_dict = {}
            # only agents that have an observation at this timestep should act
            for ts in ts_ids:
                if ts not in obs_dict:
                    continue

                obs_ts = torch.tensor(obs_dict[ts], dtype=torch.float32, device=device)
                action, logprob, _, value = agents[ts].get_action_and_value(obs_ts)
                actions_dict[ts] = action.item()

                # Zapisujemy pamięć tylko dla aktywnych agentów
                memories[ts]["obs"].append(obs_ts)
                memories[ts]["actions"].append(action)
                memories[ts]["logprobs"].append(logprob)
                memories[ts]["values"].append(value)

            # Pchanie całego środowiska i sprawdzanie, czy opłacało się zmienić to światło
            res = env.step(actions_dict)
            if len(res) == 5:
                next_obs_dict, rewards_dict, dones_dict, _, _ = res
            else:
                next_obs_dict, rewards_dict, dones_dict, _ = res

            for ts in ts_ids:
                # rewards_dict only contains keys for agents that acted
                if ts in rewards_dict:
                    memories[ts]["rewards"].append(rewards_dict[ts])

                d = (
                    dones_dict
                    if not isinstance(dones_dict, dict)
                    else dones_dict.get(ts, False)
                )
                memories[ts]["dones"].append(1.0 if d else 0.0)

            obs_dict = next_obs_dict
            if isinstance(dones_dict, dict) and all(dones_dict.values()):
                obs_dict = env.reset()
                if isinstance(obs_dict, tuple):
                    obs_dict = obs_dict[0]

        # 3. Odpalamy zaktualizowanie mózgów na tym co zebraliśmy (Algorytm PPO)
        for ts in ts_ids:
            T = len(memories[ts]["obs"])
            if T == 0:
                continue

            obs_t = torch.stack(memories[ts]["obs"]).to(device)
            act_t = torch.stack(memories[ts]["actions"]).to(device)
            logp_t = torch.stack(memories[ts]["logprobs"]).to(device)
            rew_t = torch.tensor(
                memories[ts]["rewards"], dtype=torch.float32, device=device
            )
            val_t = torch.stack(memories[ts]["values"]).to(device)
            don_t = torch.tensor(
                memories[ts]["dones"], dtype=torch.float32, device=device
            )

            # Wliczanie "Korzyści" - Advantage (czy zrobiliśmy lepiej niż się spodziewaliśmy)
            advantages = torch.zeros_like(rew_t)
            lastgaelam = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - don_t[t]
                    with torch.no_grad():
                        next_obs = torch.tensor(
                            obs_dict.get(ts, memories[ts]["obs"][-1]),
                            dtype=torch.float32,
                            device=device,
                        )
                        nextvalues = agents[ts].critic(next_obs).squeeze(-1)
                else:
                    nextnonterminal = 1.0 - don_t[t]
                    nextvalues = val_t[t + 1]

                delta = rew_t[t] + GAMMA * nextvalues * nextnonterminal - val_t[t]
                advantages[t] = lastgaelam = (
                    delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                )

            returns = advantages + val_t

            # Właściwa poprawka wag z klipowaniem
            for epoch in range(EPOCHS):
                _, newlogprob, entropy, newvalue = agents[ts].get_action_and_value(
                    obs_t, act_t
                )

                logratio = newlogprob - logp_t.detach()
                ratio = logratio.exp()

                # Funkcje PPO - bierzemy mniejszy (zdrowszy) gradient
                adv_detached = advantages.detach()
                pg_loss1 = -adv_detached * ratio
                pg_loss2 = -adv_detached * torch.clamp(
                    ratio, 1 - CLIP_FRAC, 1 + CLIP_FRAC
                )
                actor_loss = torch.min(pg_loss1, pg_loss2).mean()

                critic_loss = 0.5 * ((newvalue - returns.detach()) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = actor_loss - 0.01 * entropy_loss + 0.5 * critic_loss

                optimizers[ts].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agents[ts].parameters(), 0.5)
                optimizers[ts].step()

        # Average reward across agents that produced rollouts this epoch
        agent_means = []
        for ts in ts_ids:
            if len(memories[ts]["rewards"]) > 0:
                agent_means.append(float(np.mean(memories[ts]["rewards"])))
        if len(agent_means) > 0:
            print(
                f"Średnia nagroda na agenta (średnio po agentach): {np.mean(agent_means):.3f}"
            )
        else:
            print("Brak danych nagród w tej epoce.")

    # ===== ZAPISYWANIE MODELU PO ZAKOŃCZENIU TRENINGU =====
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "ppo_models_weights.pth")

    torch.save({ts: agents[ts].state_dict() for ts in ts_ids}, model_path)
    print(f"\nTrening zakończony pomyślnie. Wagi zapisano w {model_path}.")


if __name__ == "__main__":
    print("Mój Dedykowany PPO PyTorch wędruje do pamięci operacyjnej...")
    train()
