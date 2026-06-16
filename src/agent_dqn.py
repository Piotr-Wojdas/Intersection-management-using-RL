import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import src.setup_sumo 
from src.Sumo.sumo_rl import SumoEnvironment
from src.params import (
    ENV_CONFIG, NET_FILE, ROUTE_FILE, GLOBAL_SEED, GAMMA, REWARD_SCALE,
    DQN_TOTAL_TIMESTEPS, DQN_BUFFER_CAPACITY, DQN_BATCH_SIZE, DQN_LEARNING_RATE,
    DQN_TARGET_UPDATE_FREQ, DQN_TRAIN_FREQ, DQN_START_LEARNING,
    DQN_EPSILON_START, DQN_EPSILON_END, DQN_EPSILON_DECAY_STEPS,
    build_training_artifacts
)
from src.utils import env_reset, env_step, make_log_fn, obs_to_tensor, resolve_device


class DQNAgent(nn.Module):
    """Sieć Q-Network wyliczająca wartość dla każdej możliwej akcji."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] = [256, 128]):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor):
        return self.net(obs)

    @torch.no_grad()
    def get_greedy_action(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> int:
        """Zwraca akcję o najwyższym Q-value."""
        q_values = self.forward(obs)
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, -1e9)
        return int(torch.argmax(q_values).item())


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        return (
            torch.stack(obs).to(self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(-1),
            torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1),
            torch.stack(next_obs).to(self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)


def build_env(use_gui: bool = False, sumo_seed=None, **overrides):
    env_args = dict(ENV_CONFIG)
    env_args.setdefault("net_file", NET_FILE)
    env_args.setdefault("route_file", ROUTE_FILE)
    env_args["use_gui"] = use_gui
    if sumo_seed is not None:
        env_args["sumo_seed"] = sumo_seed
    env_args.update(overrides)
    return SumoEnvironment(**env_args)


def _action_mask_tensor(env, ts: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(env.action_masks(ts), dtype=torch.bool, device=device)


def select_action(
    net: DQNAgent, obs: torch.Tensor, epsilon: float, 
    action_mask: torch.Tensor, act_dim: int
) -> int:
    if random.random() < epsilon:
        valid_actions = torch.nonzero(action_mask).view(-1).tolist()
        return random.choice(valid_actions) if valid_actions else 0
    else:
        return net.get_greedy_action(obs, action_mask)


def train():
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    device = resolve_device()
    artifacts = build_training_artifacts()
    run_id = int(artifacts["run_id"])

    log_file_path = artifacts["log_file"]
    weights_file_path = artifacts["weights_file"]
    os.makedirs(os.path.dirname(weights_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    log_file = open(log_file_path, "w", encoding="utf-8")
    log = make_log_fn(log_file)

    env = build_env(use_gui=False)
    ts_ids = env.ts_ids

    obs_dims = {ts: int(env.observation_spaces(ts).shape[0]) for ts in ts_ids}
    act_dims = {ts: int(env.action_spaces(ts).n) for ts in ts_ids}

    policy_nets: dict[str, DQNAgent] = {}
    target_nets: dict[str, DQNAgent] = {}
    optimizers: dict[str, optim.Optimizer] = {}
    buffers: dict[str, ReplayBuffer] = {}

    log(f"Rozpoczynam trening DQN (run {run_id})")
    
    for ts in ts_ids:
        policy_nets[ts] = DQNAgent(obs_dims[ts], act_dims[ts]).to(device)
        
        target_nets[ts] = DQNAgent(obs_dims[ts], act_dims[ts]).to(device)
        target_nets[ts].load_state_dict(policy_nets[ts].state_dict())
        target_nets[ts].eval()

        optimizers[ts] = optim.Adam(policy_nets[ts].parameters(), lr=DQN_LEARNING_RATE)
        buffers[ts] = ReplayBuffer(DQN_BUFFER_CAPACITY, device)

    obs_dict = env_reset(env)
    episode_rewards = {ts: 0.0 for ts in ts_ids}
    episode_count = 0

    try:
        for step in range(1, DQN_TOTAL_TIMESTEPS + 1):
            
            epsilon = max(
                DQN_EPSILON_END, 
                DQN_EPSILON_START - (step / DQN_EPSILON_DECAY_STEPS) * (DQN_EPSILON_START - DQN_EPSILON_END)
            )

            active_ts = []
            actions_dict = {}
            obs_tensor_dict = {}
            
            for ts in ts_ids:
                if ts not in obs_dict:
                    continue
                active_ts.append(ts)
                obs_ts = obs_to_tensor(obs_dict[ts], obs_dims[ts], device)
                obs_tensor_dict[ts] = obs_ts
                mask_ts = _action_mask_tensor(env, ts, device)
                
                actions_dict[ts] = select_action(
                    policy_nets[ts], obs_ts, epsilon, mask_ts, act_dims[ts]
                )

            next_obs_dict, rewards_dict, dones_dict, _ = env_step(env, actions_dict)
            episode_ended = isinstance(dones_dict, dict) and dones_dict.get("__all__", False)

            for ts in active_ts:
                reward_value = float(rewards_dict.get(ts, 0.0))
                reward_value = float(np.clip(reward_value, -1000.0, 1000.0))
                episode_rewards[ts] += reward_value
                
                scaled_reward = reward_value * REWARD_SCALE
                
                next_obs = next_obs_dict.get(ts)
                if next_obs is not None:
                    next_obs_ts = obs_to_tensor(next_obs, obs_dims[ts], device)
                    done_flag = 1.0 if episode_ended else 0.0
                    buffers[ts].push(obs_tensor_dict[ts], actions_dict[ts], scaled_reward, next_obs_ts, done_flag)

            obs_dict = next_obs_dict
            
            if episode_ended:
                episode_count += 1
                if episode_count % 5 == 0:
                    avg_rew = np.mean(list(episode_rewards.values()))
                    log(f"Krok {step}/{DQN_TOTAL_TIMESTEPS} | Epizod {episode_count} | Śr. Nagroda/Agenta: {avg_rew:.2f} | Epsilon: {epsilon:.3f}")
                
                obs_dict = env_reset(env)
                episode_rewards = {ts: 0.0 for ts in ts_ids}

            if step > DQN_START_LEARNING and step % DQN_TRAIN_FREQ == 0:
                for ts in ts_ids:
                    if len(buffers[ts]) < DQN_BATCH_SIZE:
                        continue
                    
                    b_obs, b_act, b_rew, b_next_obs, b_dones = buffers[ts].sample(DQN_BATCH_SIZE)
                    
                    q_values = policy_nets[ts](b_obs)
                    q_values = q_values.gather(1, b_act)
                    
                    with torch.no_grad():
                        next_q_values = target_nets[ts](b_next_obs)

                        max_next_q_values = next_q_values.max(1, keepdim=True)[0]
                        target_q_values = b_rew + (GAMMA * max_next_q_values * (1 - b_dones))

                    loss = nn.MSELoss()(q_values, target_q_values)
                    
                    optimizers[ts].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_nets[ts].parameters(), 1.0)
                    optimizers[ts].step()

            if step % DQN_TARGET_UPDATE_FREQ == 0:
                for ts in ts_ids:
                    target_nets[ts].load_state_dict(policy_nets[ts].state_dict())

        ckpt = {
            "model_state_dict": {ts: policy_nets[ts].state_dict() for ts in ts_ids},
            "obs_dims": obs_dims,
            "act_dims": act_dims,
            "ts_ids": ts_ids,
        }
        torch.save(ckpt, weights_file_path)
        log(f"\nTrening zakończony pomyślnie. Wagi zapisano w {weights_file_path}.")

    except KeyboardInterrupt:
        log("\nPrzerwano przez użytkownika. Symulacja zatrzymana.")
    finally:
        env.close()
        log_file.close()

if __name__ == "__main__":
    train()