import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from src.Sumo.sumo_rl import SumoEnvironment
from src.params import (
    CLIP_FRAC,
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
    build_training_artifacts,
)
from src.utils import pad_observation, resolve_device


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight init — standard for PPO stability."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SharedPPOAgent(nn.Module):
    """Independent PPO actor-critic for one traffic signal.

    One-hot agent embedding was removed: each agent has its own separate network,
    so the embedding was always a constant [1.0] and wasted an input dimension.
    Architecture uses ELU + orthogonal init instead of Tanh + default init.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        if hidden_dims is None:
            hidden_dims = list(PPO_HIDDEN_DIMS)

        layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(_layer_init(nn.Linear(last_dim, hidden_dim)))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.actor = _layer_init(nn.Linear(last_dim, act_dim), std=0.01)
        self.critic = _layer_init(nn.Linear(last_dim, 1), std=1.0)

    def _forward(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        latent = self.shared(obs)
        logits = self.actor(latent)
        value = self.critic(latent).squeeze(-1)
        return logits, value

    def _mask_logits(self, logits: torch.Tensor, valid_action_dim):
        """Mask logits for actions beyond valid_action_dim to -1e9."""
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

    def _sanitize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return torch.clamp(logits, -20.0, 20.0)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        valid_action_dim=None,
    ):
        logits, value = self._forward(obs)
        logits = self._sanitize_logits(logits)
        logits = self._mask_logits(logits, valid_action_dim)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def get_value(self, obs: torch.Tensor):
        _, value = self._forward(obs)
        return value


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


def _run_single_eval(
    agents: dict,
    device: torch.device,
    ts_ids: list[str],
    act_dims: dict[str, int],
    obs_dims: dict[str, int],
    seed: int,
) -> float:
    """Run one greedy episode and return the mean per-agent reward sum."""
    eval_env = build_env(use_gui=False, sumo_seed=seed)
    try:
        obs_dict = eval_env.reset(seed=seed)
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


def evaluate_agent(
    agents: dict,
    device: torch.device,
    ts_ids: list[str],
    act_dims: dict[str, int],
    obs_dims: dict[str, int],
    eval_seeds,
) -> float:
    """Average greedy eval score over one or more seeds for a stable signal."""
    if isinstance(eval_seeds, int):
        eval_seeds = [eval_seeds]
    scores = [
        _run_single_eval(agents, device, ts_ids, act_dims, obs_dims, s)
        for s in eval_seeds
    ]
    return float(np.mean(scores))


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

        obs_dims = {ts: env.observation_spaces(ts).shape[0] for ts in ts_ids}
        act_dims = {ts: env.action_spaces(ts).n for ts in ts_ids}

        agents: dict[str, SharedPPOAgent] = {}
        optimizers: dict[str, optim.Optimizer] = {}
        for ts in ts_ids:
            agents[ts] = SharedPPOAgent(obs_dims[ts], act_dims[ts]).to(device)
            optimizers[ts] = optim.Adam(agents[ts].parameters(), lr=LEARNING_RATE)

        log(f"Stworzono niezależne modele PPO dla {len(ts_ids)} skrzyżowań (obs_dims={obs_dims}).")

        best_eval_score = -float("inf")

        obs_dict = env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]

        for update in range(1, NUM_UPDATES + 1):
            progress = _schedule_progress(update, NUM_UPDATES)
            current_lr = _scheduled_lr(progress)
            current_entropy_coef = _scheduled_entropy_coef(progress)
            for opt in optimizers.values():
                _set_optimizer_lr(opt, current_lr)

            log(
                f"\nUruchamiam epokę uczenia #{update} | lr={current_lr:.6f} | entropy_coef={current_entropy_coef:.6f}"
            )

            # Reset at the start of every rollout so each update sees exactly
            # one full episode from t=0. This eliminates the episode-position
            # variance that made training rewards jump wildly between updates.
            obs_dict = env.reset()
            if isinstance(obs_dict, tuple):
                obs_dict = obs_dict[0]

            memories = {
                ts: {
                    "obs": [],
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
                # Track only agents that have an observation this step so that
                # all per-agent memory lists stay the same length (fixes length
                # mismatch when rewards/dones were appended unconditionally).
                active_ts = []
                actions_dict = {}

                for ts in ts_ids:
                    if ts not in obs_dict:
                        continue
                    active_ts.append(ts)

                    obs_ts = torch.tensor(
                        obs_dict[ts], dtype=torch.float32, device=device
                    )
                    obs_ts = pad_observation(obs_ts, obs_dims[ts])
                    obs_ts = torch.nan_to_num(obs_ts, nan=0.0, posinf=1.0, neginf=-1.0)
                    valid_act_dim = act_dims[ts]

                    action, logprob, _, value = agents[ts].get_action_and_value(
                        obs_ts,
                        valid_action_dim=valid_act_dim,
                    )
                    actions_dict[ts] = int(action.item())

                    memories[ts]["obs"].append(obs_ts.detach())
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

                # SUMO only sets __all__=True; individual agent dones are always
                # False. We must propagate the episode boundary to all active
                # agents so GAE does not bootstrap across episode boundaries.
                episode_ended = (
                    isinstance(dones_dict, dict) and dones_dict.get("__all__", False)
                )

                for ts in active_ts:
                    if isinstance(rewards_dict, dict) and ts in rewards_dict:
                        reward_value = float(rewards_dict[ts])
                        if not np.isfinite(reward_value):
                            reward_value = 0.0
                        reward_value = float(np.clip(reward_value, -1000.0, 1000.0))
                    else:
                        reward_value = 0.0
                    memories[ts]["rewards"].append(reward_value)
                    epoch_reward_sum[ts] += reward_value
                    memories[ts]["dones"].append(1.0 if episode_ended else 0.0)

                obs_dict = next_obs_dict
                if episode_ended:
                    obs_dict = env.reset()
                    if isinstance(obs_dict, tuple):
                        obs_dict = obs_dict[0]

            any_updates = False
            for ts in ts_ids:
                T = len(memories[ts]["obs"])
                if T == 0:
                    continue
                any_updates = True

                obs_t = torch.stack(memories[ts]["obs"]).to(device)
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

                # GAE advantage computation.
                # When don_t[t]=1 (episode ended) nextnonterminal=0 prevents
                # bootstrapping across the episode boundary.
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
                        next_obs = pad_observation(next_obs, obs_dims[ts])
                        with torch.no_grad():
                            nextvalue = agents[ts].get_value(next_obs).view(-1)[0]
                    else:
                        nextvalue = val_t[t + 1]

                    delta = rew_t[t] + GAMMA * nextvalue * nextnonterminal - val_t[t]
                    lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam

                returns = advantages + val_t
                advantages = torch.nan_to_num(
                    advantages, nan=0.0, posinf=1000.0, neginf=-1000.0
                )
                returns = torch.nan_to_num(
                    returns, nan=0.0, posinf=1000.0, neginf=-1000.0
                )

                adv_b = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )

                batch_size = obs_t.shape[0]
                minibatch_size = min(PPO_MINIBATCH_SIZE, batch_size)
                for _epoch in range(EPOCHS):
                    permutation = torch.randperm(batch_size, device=device)
                    for start in range(0, batch_size, minibatch_size):
                        idx = permutation[start : start + minibatch_size]
                        mb_obs = obs_t[idx]
                        mb_act = act_t[idx]
                        mb_act_dim = act_dim_t[idx]
                        mb_logp_old = logp_t[idx]
                        mb_adv = adv_b[idx]
                        mb_ret = returns[idx]
                        mb_old_val = val_t[idx]

                        _, newlogprob, entropy, newvalue = agents[ts].get_action_and_value(
                            mb_obs,
                            action=mb_act,
                            valid_action_dim=mb_act_dim,
                        )

                        ratio = (newlogprob - mb_logp_old.detach()).exp()
                        pg_loss1 = -mb_adv.detach() * ratio
                        pg_loss2 = -mb_adv.detach() * torch.clamp(
                            ratio, 1 - CLIP_FRAC, 1 + CLIP_FRAC
                        )
                        actor_loss = torch.min(pg_loss1, pg_loss2).mean()
                        # Clipped value loss prevents large critic updates (PPO standard).
                        v_clipped = mb_old_val + torch.clamp(
                            newvalue - mb_old_val, -CLIP_FRAC, CLIP_FRAC
                        )
                        critic_loss = 0.5 * torch.max(
                            (newvalue - mb_ret.detach()) ** 2,
                            (v_clipped - mb_ret.detach()) ** 2,
                        ).mean()
                        entropy_loss = entropy.mean()

                        loss = (
                            actor_loss
                            - current_entropy_coef * entropy_loss
                            + PPO_VALUE_COEF * critic_loss
                        )

                        optimizers[ts].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agents[ts].parameters(), PPO_MAX_GRAD_NORM)
                        optimizers[ts].step()

            if not any_updates:
                log("Brak danych do aktualizacji w tej epoce.")
                continue

            per_agent_summary = ", ".join(
                f"{ts}={epoch_reward_sum[ts]:.3f}" for ts in ts_ids
            )
            log(f"Nagroda per agent: {per_agent_summary}")
            log(
                f"Srednia nagroda na agenta: {_aggregate_policy_reward(epoch_reward_sum):.3f}"
            )

            if update % TRAIN_EVAL_EVERY_UPDATES == 0:
                eval_score = evaluate_agent(
                    agents,
                    device,
                    ts_ids,
                    act_dims,
                    obs_dims,
                    TRAIN_EVAL_SEED,
                )
                log(f"Ocena greedy: {eval_score:.3f}")
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_checkpoint = {
                        "model_state_dict": {ts: agents[ts].state_dict() for ts in ts_ids},
                        "obs_dims": obs_dims,
                        "act_dims": act_dims,
                        "ts_ids": ts_ids,
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
            "model_state_dict": {ts: agents[ts].state_dict() for ts in ts_ids},
            "obs_dims": obs_dims,
            "act_dims": act_dims,
            "ts_ids": ts_ids,
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
    print("Startuję niezależne PPO (oddzielne polityki) dla mapy city_map_2...")
    train()
