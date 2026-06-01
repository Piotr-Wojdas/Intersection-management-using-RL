import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from src.Sumo.sumo_rl import SumoEnvironment
from src.params import (
    CLIP_FRAC,
    DEVICE_OVERRIDE,
    ENV_CONFIG,
    EPOCHS,
    GAMMA,
    GAE_LAMBDA,
    LEARNING_RATE,
    NET_FILE,
    NUM_UPDATES,
    PPO_ENTROPY_COEF,
    PPO_ENTROPY_FINAL_FRAC,
    PPO_HIDDEN_DIMS,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH_SIZE,
    PPO_VALUE_COEF,
    ROUTE_FILE,
    ROLLOUT_STEPS,
    TRAIN_EVAL_EVERY_UPDATES,
    TRAIN_EVAL_SEED,
    USE_CUDA_IF_AVAILABLE,
    build_training_artifacts,
)


class SharedPPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        if hidden_dims is None:
            hidden_dims = list(PPO_HIDDEN_DIMS)

        input_dim = obs_dim + num_agents
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(last_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            last_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(last_dim, act_dim)
        self.critic = nn.Linear(last_dim, 1)

    def _forward(self, obs: torch.Tensor, ts_idx: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if ts_idx.dim() == 0:
            ts_idx = ts_idx.unsqueeze(0)
        ts_embed = F.one_hot(ts_idx.long(), num_classes=self.num_agents).float()
        x = torch.cat([obs, ts_embed], dim=-1)
        latent = self.shared(x)
        logits = self.actor(latent)
        value = self.critic(latent).squeeze(-1)
        return logits, value

    def _mask_logits(self, logits: torch.Tensor, valid_action_dim):
        if valid_action_dim is None:
            return logits

        squeeze_back = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            squeeze_back = True

        if not torch.is_tensor(valid_action_dim):
            valid_action_dim = torch.tensor(valid_action_dim, device=logits.device)
        if valid_action_dim.dim() == 0:
            valid_action_dim = valid_action_dim.unsqueeze(0)
        valid_action_dim = valid_action_dim.long()

        action_ids = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
        mask = action_ids >= valid_action_dim.unsqueeze(-1)
        logits = logits.masked_fill(mask, -1e9)

        if squeeze_back:
            logits = logits.squeeze(0)
        return logits

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        ts_idx: torch.Tensor,
        action: torch.Tensor | None = None,
        valid_action_dim=None,
    ):
        logits, value = self._forward(obs, ts_idx)
        logits = self._mask_logits(logits, valid_action_dim)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value

    def get_value(self, obs: torch.Tensor, ts_idx: torch.Tensor):
        _, value = self._forward(obs, ts_idx)
        return value


def resolve_device():
    if DEVICE_OVERRIDE is not None:
        return torch.device(DEVICE_OVERRIDE)
    if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pad_observation(obs: torch.Tensor, target_dim: int):
    if obs.dim() != 1:
        obs = obs.view(-1)
    if obs.shape[0] == target_dim:
        return obs
    padded = torch.zeros(target_dim, dtype=obs.dtype, device=obs.device)
    padded[: obs.shape[0]] = obs
    return padded


def build_env(use_gui: bool = False, sumo_seed=None):
    env_args = dict(ENV_CONFIG)
    env_args.setdefault("net_file", NET_FILE)
    env_args.setdefault("route_file", ROUTE_FILE)
    env_args["use_gui"] = use_gui
    if sumo_seed is not None:
        env_args["sumo_seed"] = sumo_seed
    return SumoEnvironment(**env_args)


def _set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _schedule_progress(update_index: int, total_updates: int) -> float:
    if total_updates <= 1:
        return 1.0
    return (update_index - 1) / (total_updates - 1)


def _scheduled_lr(progress: float) -> float:
    return LEARNING_RATE * (1.0 - 0.9 * progress)


def _scheduled_entropy_coef(progress: float) -> float:
    final_frac = PPO_ENTROPY_FINAL_FRAC
    return PPO_ENTROPY_COEF * (final_frac + (1.0 - final_frac) * (1.0 - progress))


def _aggregate_policy_reward(reward_sum: dict[str, float]) -> float:
    values = list(reward_sum.values())
    if not values:
        return 0.0
    return float(np.mean(values))


def evaluate_agent(
    agent: SharedPPOAgent,
    device: torch.device,
    ts_ids: list[str],
    ts_to_idx: dict[str, int],
    act_dims: dict[str, int],
    max_obs_dim: int,
    eval_seed,
) -> float:
    eval_env = build_env(use_gui=False, sumo_seed=eval_seed)
    try:
        obs_dict = eval_env.reset(seed=eval_seed)
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]

        reward_sum = {ts: 0.0 for ts in ts_ids}
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

            res = eval_env.step(actions_dict)
            if len(res) == 5:
                obs_dict, rewards_dict, terminated, truncated, _info = res
                dones_dict = {"__all__": bool(terminated or truncated)}
            else:
                obs_dict, rewards_dict, dones_dict, _info = res

            if isinstance(rewards_dict, dict):
                for ts in ts_ids:
                    reward_sum[ts] += float(rewards_dict.get(ts, 0.0))

            if isinstance(dones_dict, dict):
                done = bool(dones_dict.get("__all__", False))
            else:
                done = bool(dones_dict)

        return _aggregate_policy_reward(reward_sum)
    finally:
        eval_env.close()


def train():
    device = resolve_device()
    artifacts = build_training_artifacts()
    log_file_path = artifacts["log_file"]
    weights_file_path = artifacts["weights_file"]
    run_id = int(artifacts["run_id"])
    best_weights_file_path = os.path.join(
        os.path.dirname(weights_file_path), f"ppo_models_weights_{run_id}_best.pth"
    )

    os.makedirs(os.path.dirname(weights_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    log_file = open(log_file_path, "w", encoding="utf-8")

    def log(message=""):
        print(message)
        print(message, file=log_file, flush=True)

    env = None
    try:
        log(f"Plik wag: {weights_file_path}")
        log(f"Plik najlepszego modelu: {best_weights_file_path}")
        log(f"Plik logu: {log_file_path}")
        log(f"Używam urządzenia: {device}")

        env = build_env(use_gui=False)
        ts_ids = env.ts_ids
        ts_to_idx = {ts: idx for idx, ts in enumerate(ts_ids)}

        obs_dims = {ts: env.observation_spaces(ts).shape[0] for ts in ts_ids}
        act_dims = {ts: env.action_spaces(ts).n for ts in ts_ids}
        max_obs_dim = max(obs_dims.values())
        max_act_dim = max(act_dims.values())

        agent = SharedPPOAgent(max_obs_dim, max_act_dim, len(ts_ids)).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

        log(
            f"Stworzono współdzielony model PPO dla {len(ts_ids)} skrzyżowań (obs={max_obs_dim}, act={max_act_dim})."
        )

        eval_seed = TRAIN_EVAL_SEED
        best_eval_score = -float("inf")

        obs_dict = env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]

        for update in range(1, NUM_UPDATES + 1):
            progress = _schedule_progress(update, NUM_UPDATES)
            current_lr = _scheduled_lr(progress)
            current_entropy_coef = _scheduled_entropy_coef(progress)
            _set_optimizer_lr(optimizer, current_lr)

            log(
                f"\nUruchamiam epokę uczenia #{update} | lr={current_lr:.6f} | entropy_coef={current_entropy_coef:.6f}"
            )

            memories = {
                ts: {
                    "obs": [],
                    "ts_idx": [],
                    "act_dim": [],
                    "actions": [],
                    "logprobs": [],
                    "rewards": [],
                    "values": [],
                    "dones": [],
                }
                for ts in ts_ids
            }
            epoch_reward_sum = {ts: 0.0 for ts in ts_ids}

            for _step in range(ROLLOUT_STEPS):
                actions_dict = {}

                for ts in ts_ids:
                    if ts not in obs_dict:
                        continue

                    obs_ts = torch.tensor(
                        obs_dict[ts], dtype=torch.float32, device=device
                    )
                    obs_ts = pad_observation(obs_ts, max_obs_dim)
                    ts_idx = torch.tensor(
                        ts_to_idx[ts], dtype=torch.long, device=device
                    )
                    valid_act_dim = act_dims[ts]

                    action, logprob, _, value = agent.get_action_and_value(
                        obs_ts,
                        ts_idx,
                        valid_action_dim=valid_act_dim,
                    )
                    actions_dict[ts] = int(action.item())

                    memories[ts]["obs"].append(obs_ts.detach())
                    memories[ts]["ts_idx"].append(ts_idx.detach())
                    memories[ts]["act_dim"].append(valid_act_dim)
                    memories[ts]["actions"].append(action.detach())
                    memories[ts]["logprobs"].append(logprob.detach())
                    memories[ts]["values"].append(value.detach().squeeze(-1))

                res = env.step(actions_dict)
                if len(res) == 5:
                    next_obs_dict, rewards_dict, terminated, truncated, _info = res
                    dones_dict = {"__all__": bool(terminated or truncated)}
                else:
                    next_obs_dict, rewards_dict, dones_dict, _info = res

                for ts in ts_ids:
                    if isinstance(rewards_dict, dict) and ts in rewards_dict:
                        reward_value = float(rewards_dict[ts])
                        memories[ts]["rewards"].append(reward_value)
                        epoch_reward_sum[ts] += reward_value
                    d = (
                        dones_dict.get(ts, False)
                        if isinstance(dones_dict, dict)
                        else bool(dones_dict)
                    )
                    memories[ts]["dones"].append(1.0 if d else 0.0)

                obs_dict = next_obs_dict
                if isinstance(dones_dict, dict) and dones_dict.get("__all__", False):
                    obs_dict = env.reset()
                    if isinstance(obs_dict, tuple):
                        obs_dict = obs_dict[0]

            batch_obs = []
            batch_ts_idx = []
            batch_act_dim = []
            batch_actions = []
            batch_logprobs = []
            batch_advantages = []
            batch_returns = []

            for ts in ts_ids:
                T = len(memories[ts]["obs"])
                if T == 0:
                    continue

                obs_t = torch.stack(memories[ts]["obs"]).to(device)
                ts_idx_t = torch.tensor(
                    memories[ts]["ts_idx"], dtype=torch.long, device=device
                )
                act_dim_t = torch.tensor(
                    memories[ts]["act_dim"], dtype=torch.long, device=device
                )
                act_t = torch.stack(memories[ts]["actions"]).to(device)
                logp_t = torch.stack(memories[ts]["logprobs"]).to(device)
                rew_t = torch.tensor(
                    memories[ts]["rewards"], dtype=torch.float32, device=device
                )
                val_t = torch.stack(memories[ts]["values"]).to(device).view(-1)
                don_t = torch.tensor(
                    memories[ts]["dones"], dtype=torch.float32, device=device
                )

                advantages = torch.zeros_like(rew_t)
                lastgaelam = 0.0
                for t in reversed(range(T)):
                    nextnonterminal = 1.0 - don_t[t]
                    if t == T - 1:
                        next_obs_source = obs_dict.get(
                            ts, memories[ts]["obs"][-1].detach().cpu().numpy()
                        )
                        next_obs = torch.tensor(
                            next_obs_source, dtype=torch.float32, device=device
                        )
                        next_obs = pad_observation(next_obs, max_obs_dim)
                        with torch.no_grad():
                            nextvalue = agent.get_value(next_obs, ts_idx_t[t]).view(-1)[
                                0
                            ]
                    else:
                        nextvalue = val_t[t + 1]

                    delta = rew_t[t] + GAMMA * nextvalue * nextnonterminal - val_t[t]
                    lastgaelam = (
                        delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                    )
                    advantages[t] = lastgaelam

                returns = advantages + val_t
                batch_obs.append(obs_t)
                batch_ts_idx.append(ts_idx_t)
                batch_act_dim.append(act_dim_t)
                batch_actions.append(act_t)
                batch_logprobs.append(logp_t)
                batch_advantages.append(advantages)
                batch_returns.append(returns)

            if len(batch_obs) == 0:
                log("Brak danych do aktualizacji w tej epoce.")
                continue

            obs_b = torch.cat(batch_obs, dim=0)
            ts_b = torch.cat(batch_ts_idx, dim=0)
            act_dim_b = torch.cat(batch_act_dim, dim=0)
            act_b = torch.cat(batch_actions, dim=0)
            logp_old_b = torch.cat(batch_logprobs, dim=0)
            adv_b = torch.cat(batch_advantages, dim=0)
            ret_b = torch.cat(batch_returns, dim=0)

            adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

            batch_size = obs_b.shape[0]
            minibatch_size = min(PPO_MINIBATCH_SIZE, batch_size)
            for _epoch in range(EPOCHS):
                permutation = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    idx = permutation[start : start + minibatch_size]
                    mb_obs = obs_b[idx]
                    mb_ts = ts_b[idx]
                    mb_act = act_b[idx]
                    mb_act_dim = act_dim_b[idx]
                    mb_logp_old = logp_old_b[idx]
                    mb_adv = adv_b[idx]
                    mb_ret = ret_b[idx]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        mb_obs,
                        mb_ts,
                        action=mb_act,
                        valid_action_dim=mb_act_dim,
                    )

                    ratio = (newlogprob - mb_logp_old.detach()).exp()
                    pg_loss1 = -mb_adv.detach() * ratio
                    pg_loss2 = -mb_adv.detach() * torch.clamp(
                        ratio, 1 - CLIP_FRAC, 1 + CLIP_FRAC
                    )
                    actor_loss = torch.min(pg_loss1, pg_loss2).mean()
                    critic_loss = 0.5 * ((newvalue - mb_ret.detach()) ** 2).mean()
                    entropy_loss = entropy.mean()

                    loss = (
                        actor_loss
                        - current_entropy_coef * entropy_loss
                        + PPO_VALUE_COEF * critic_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), PPO_MAX_GRAD_NORM)
                    optimizer.step()

            per_agent_summary = ", ".join(
                f"{ts}={epoch_reward_sum[ts]:.3f}" for ts in ts_ids
            )
            log(f"Nagroda per agent: {per_agent_summary}")
            log(
                f"Srednia nagroda na agenta: {_aggregate_policy_reward(epoch_reward_sum):.3f}"
            )

            if update % TRAIN_EVAL_EVERY_UPDATES == 0:
                eval_score = evaluate_agent(
                    agent,
                    device,
                    ts_ids,
                    ts_to_idx,
                    act_dims,
                    max_obs_dim,
                    eval_seed,
                )
                log(f"Ocena greedy: {eval_score:.3f}")
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_checkpoint = {
                        "model_state_dict": agent.state_dict(),
                        "obs_dim": max_obs_dim,
                        "act_dim": max_act_dim,
                        "ts_ids": ts_ids,
                        "ts_to_idx": ts_to_idx,
                        "map_net_file": NET_FILE,
                        "map_route_file": ROUTE_FILE,
                        "best_eval_score": best_eval_score,
                        "update": update,
                    }
                    torch.save(best_checkpoint, best_weights_file_path)
                    log(
                        f"Nowy najlepszy model: {best_weights_file_path} (score={best_eval_score:.3f})"
                    )

        checkpoint = {
            "model_state_dict": agent.state_dict(),
            "obs_dim": max_obs_dim,
            "act_dim": max_act_dim,
            "ts_ids": ts_ids,
            "ts_to_idx": ts_to_idx,
            "map_net_file": NET_FILE,
            "map_route_file": ROUTE_FILE,
            "best_eval_score": best_eval_score,
        }
        torch.save(checkpoint, weights_file_path)
        log(f"\nTrening zakończony pomyślnie. Wagi zapisano w {weights_file_path}.")
        if os.path.exists(best_weights_file_path):
            log(f"Najlepszy checkpoint zapisano w {best_weights_file_path}")
    finally:
        if env is not None:
            env.close()
        log_file.close()


if __name__ == "__main__":
    print("Startuję współdzielony PPO dla mapy city_map_2...")
    train()
