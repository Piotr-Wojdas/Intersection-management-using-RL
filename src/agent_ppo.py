import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import src.setup_sumo  # noqa: F401  (sets SUMO_HOME before any TraCI/sumolib use)
from src.Sumo.sumo_rl import SumoEnvironment
from src.params import (
    CLIP_FRAC,
    ENV_CONFIG,
    EPOCHS,
    GAMMA,
    GAE_LAMBDA,
    GLOBAL_SEED,
    LEARNING_RATE,
    NET_FILE,
    NUM_UPDATES,
    PPO_ENTROPY_COEF,
    PPO_ENTROPY_FINAL_FRAC,
    PPO_HIDDEN_DIMS,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH_SIZE,
    PPO_VALUE_COEF,
    RESUME_SAVE_EVERY_UPDATES,
    REWARD_SCALE,
    ROUTE_FILE,
    ROLLOUT_STEPS,
    TRAIN_EVAL_EVERY_UPDATES,
    TRAIN_EVAL_SEED,
    build_resume_file,
    build_training_artifacts,
    get_latest_resume_run_id,
)
from src.utils import env_reset, env_step, make_log_fn, obs_to_tensor, resolve_device


def _layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    """Orthogonal weight init — standard for PPO stability."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    """Independent PPO actor-critic for one traffic signal.

    Invalid actions (holding during a forced switch, switching during
    min_green) are excluded via `action_mask`, so every action stored in the
    buffer is the action the environment actually executed.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, hidden_dims: list[int] | None = None
    ):
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

    @staticmethod
    def _mask_logits(logits: torch.Tensor, action_mask: torch.Tensor | None):
        """Set logits of masked-out (False) actions to -1e9."""
        if action_mask is None:
            return logits

        squeeze_back = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            squeeze_back = True
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        logits = logits.masked_fill(~action_mask, -1e9)

        if squeeze_back:
            logits = logits.squeeze(0)
        return logits

    @staticmethod
    def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return torch.clamp(logits, -20.0, 20.0)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ):
        logits, value = self._forward(obs)
        logits = self._sanitize_logits(logits)
        logits = self._mask_logits(logits, action_mask)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def get_value(self, obs: torch.Tensor):
        _, value = self._forward(obs)
        return value

    @torch.no_grad()
    def get_greedy_action(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> int:
        """Return the argmax action without gradient tracking."""
        logits, _ = self._forward(obs)
        logits = self._sanitize_logits(logits)
        logits = self._mask_logits(logits, action_mask)
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        return int(torch.argmax(logits).item())


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


def _build_checkpoint(
    agents: dict,
    ts_ids: list[str],
    obs_dims: dict,
    act_dims: dict,
    best_eval_score: float,
    update: int | None = None,
) -> dict:
    ckpt = {
        "model_state_dict": {ts: agents[ts].state_dict() for ts in ts_ids},
        "obs_dims": obs_dims,
        "act_dims": act_dims,
        "ts_ids": ts_ids,
        "map_net_file": NET_FILE,
        "map_route_file": ROUTE_FILE,
        "best_eval_score": best_eval_score,
    }
    if update is not None:
        ckpt["update"] = update
    return ckpt


def _build_resume_state(
    agents, optimizers, ts_ids, obs_dims, act_dims, update, best_eval_score
) -> dict:
    """Full training state needed to resume: weights, Adam momentum, RNG, progress."""
    return {
        "model_state_dict": {ts: agents[ts].state_dict() for ts in ts_ids},
        "optimizer_state_dict": {ts: optimizers[ts].state_dict() for ts in ts_ids},
        "obs_dims": obs_dims,
        "act_dims": act_dims,
        "ts_ids": ts_ids,
        "map_net_file": NET_FILE,
        "map_route_file": ROUTE_FILE,
        "best_eval_score": best_eval_score,
        "update": update,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
    }


def _save_resume_state(
    path, agents, optimizers, ts_ids, obs_dims, act_dims, update, best_eval_score
) -> None:
    """Write the resume checkpoint atomically (tmp + os.replace) so an interrupt
    mid-write cannot corrupt the previous good checkpoint."""
    state = _build_resume_state(
        agents, optimizers, ts_ids, obs_dims, act_dims, update, best_eval_score
    )
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _aggregate_policy_reward(reward_sum: dict[str, float]) -> float:
    values = list(reward_sum.values())
    if not values:
        return 0.0
    return float(np.mean(values))


def _run_single_eval(
    agents: dict,
    device: torch.device,
    ts_ids: list[str],
    obs_dims: dict[str, int],
    seed: int,
) -> tuple[float, dict]:
    """Run one greedy episode; return (mean per-agent reward sum, system metrics)."""
    eval_env = build_env(use_gui=False, sumo_seed=seed)
    try:
        obs_dict = env_reset(eval_env)
        reward_sum = {ts: 0.0 for ts in ts_ids}
        wait_means: list[float] = []
        last_info: dict = {}
        done = False

        while not done:
            actions_dict = {
                ts: agents[ts].get_greedy_action(
                    obs_to_tensor(obs_dict[ts], obs_dims[ts], device),
                    action_mask=_action_mask_tensor(eval_env, ts, device),
                )
                for ts in ts_ids
                if ts in obs_dict
            }

            obs_dict, rewards_dict, dones_dict, info = env_step(eval_env, actions_dict)

            if isinstance(rewards_dict, dict):
                for ts in ts_ids:
                    reward_sum[ts] += float(rewards_dict.get(ts, 0.0))
            if isinstance(info, dict):
                last_info = info
                if "system_mean_waiting_time" in info:
                    wait_means.append(float(info["system_mean_waiting_time"]))

            done = bool(dones_dict.get("__all__", False)) if isinstance(dones_dict, dict) else bool(dones_dict)

        metrics = {
            "mean_waiting_time": float(np.mean(wait_means)) if wait_means else 0.0,
            "arrived": float(last_info.get("system_total_arrived", 0.0)),
            "teleported": float(last_info.get("system_total_teleported", 0.0)),
            "backlogged_end": float(last_info.get("system_total_backlogged", 0.0)),
        }
        return _aggregate_policy_reward(reward_sum), metrics
    finally:
        eval_env.close()


def evaluate_agent(
    agents: dict,
    device: torch.device,
    ts_ids: list[str],
    obs_dims: dict[str, int],
    eval_seeds,
) -> tuple[float, dict]:
    """Average greedy eval score and system metrics over one or more seeds."""
    if isinstance(eval_seeds, int):
        eval_seeds = [eval_seeds]
    scores: list[float] = []
    metrics_list: list[dict] = []
    for s in eval_seeds:
        score, metrics = _run_single_eval(agents, device, ts_ids, obs_dims, s)
        scores.append(score)
        metrics_list.append(metrics)
    avg_metrics = {
        key: float(np.mean([m[key] for m in metrics_list])) for key in metrics_list[0]
    }
    return float(np.mean(scores)), avg_metrics


def train(resume_run_id: int | None = None):
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    device = resolve_device()

    resume_state = None
    if resume_run_id is not None:
        resume_file = build_resume_file(resume_run_id)
        if not os.path.exists(resume_file):
            raise FileNotFoundError(
                f"Brak pliku wznowienia dla run {resume_run_id}: {resume_file}. "
                "Wznowić można tylko trening uruchomiony z tą wersją kodu."
            )
        # weights_only=False: our own trusted file, and it also holds optimizer
        # and RNG state that strict loading would reject.
        resume_state = torch.load(
            resume_file, map_location=device, weights_only=False
        )
        run_id = int(resume_run_id)
        artifacts = build_training_artifacts(run_id=run_id)
    else:
        artifacts = build_training_artifacts()
        run_id = int(artifacts["run_id"])

    log_file_path = artifacts["log_file"]
    weights_file_path = artifacts["weights_file"]
    best_weights_file_path = os.path.join(
        os.path.dirname(weights_file_path), f"ppo_models_weights_{run_id}_best.pth"
    )
    resume_file_path = build_resume_file(run_id)

    os.makedirs(os.path.dirname(weights_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    log_file = open(
        log_file_path, "a" if resume_state is not None else "w", encoding="utf-8"
    )
    log = make_log_fn(log_file)

    env = None
    ready = False
    last_completed_update = 0
    agents = {}
    optimizers = {}
    try:
        if resume_state is not None:
            log(f"\n=== Wznowienie treningu (run {run_id}) ===")
        log(f"Plik wag: {weights_file_path}")
        log(f"Plik najlepszego modelu: {best_weights_file_path}")
        log(f"Plik punktu wznowienia: {resume_file_path}")
        log(f"Plik logu: {log_file_path}")
        log(f"Używam urządzenia: {device} | seed={GLOBAL_SEED}")

        env = build_env(use_gui=False)
        ts_ids = env.ts_ids

        obs_dims = {ts: int(env.observation_spaces(ts).shape[0]) for ts in ts_ids}
        act_dims = {ts: int(env.action_spaces(ts).n) for ts in ts_ids}

        agents: dict[str, PPOAgent] = {}
        optimizers: dict[str, optim.Optimizer] = {}
        for ts in ts_ids:
            agents[ts] = PPOAgent(obs_dims[ts], act_dims[ts]).to(device)
            optimizers[ts] = optim.Adam(agents[ts].parameters(), lr=LEARNING_RATE)

        log(
            f"Stworzono niezależne modele PPO dla {len(ts_ids)} skrzyżowań (obs_dims={obs_dims})."
        )

        best_eval_score = -float("inf")
        start_update = 1
        if resume_state is not None:
            if (
                resume_state.get("obs_dims") != obs_dims
                or resume_state.get("act_dims") != act_dims
            ):
                raise ValueError(
                    "Wymiary sieci w pliku wznowienia nie pasują do środowiska "
                    "(zmieniła się mapa lub funkcja obserwacji) — nie można wznowić."
                )
            for ts in ts_ids:
                agents[ts].load_state_dict(resume_state["model_state_dict"][ts])
                optimizers[ts].load_state_dict(resume_state["optimizer_state_dict"][ts])
            best_eval_score = float(resume_state.get("best_eval_score", -float("inf")))
            start_update = int(resume_state["update"]) + 1
            try:
                random.setstate(resume_state["rng_python"])
                np.random.set_state(resume_state["rng_numpy"])
                torch.set_rng_state(resume_state["rng_torch"].cpu())
            except Exception as exc:
                log(f"Ostrzeżenie: nie odtworzono stanu RNG ({exc}).")
            log(
                f"Wznowiono run {run_id} od epoki {start_update} "
                f"(ukończono {start_update - 1}/{NUM_UPDATES}, best={best_eval_score:.3f})."
            )

        ready = True
        last_completed_update = start_update - 1
        obs_dict = env_reset(env)

        for update in range(start_update, NUM_UPDATES + 1):
            progress = _schedule_progress(update, NUM_UPDATES)
            current_lr = _scheduled_lr(progress)
            current_entropy_coef = _scheduled_entropy_coef(progress)
            for opt in optimizers.values():
                _set_optimizer_lr(opt, current_lr)

            log(
                f"\nUruchamiam epokę uczenia #{update} | lr={current_lr:.6f} | entropy_coef={current_entropy_coef:.6f}"
            )

            memories = {
                ts: {
                    "obs": [],
                    "masks": [],
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
                # all per-agent memory lists stay the same length.
                active_ts = []
                actions_dict = {}

                for ts in ts_ids:
                    if ts not in obs_dict:
                        continue
                    active_ts.append(ts)

                    obs_ts = obs_to_tensor(obs_dict[ts], obs_dims[ts], device)
                    mask_ts = _action_mask_tensor(env, ts, device)

                    with torch.no_grad():
                        action, logprob, _, value = agents[ts].get_action_and_value(
                            obs_ts, action_mask=mask_ts
                        )
                    actions_dict[ts] = int(action.item())

                    mem = memories[ts]
                    mem["obs"].append(obs_ts)
                    mem["masks"].append(mask_ts)
                    # Stored as 0-d tensors so stacking yields flat [T] tensors;
                    # [T, 1] shapes silently broadcast log_prob to [B, B] later.
                    mem["actions"].append(action.view(()))
                    mem["logprobs"].append(logprob.view(()))
                    mem["values"].append(value.view(()))

                next_obs_dict, rewards_dict, dones_dict, _info = env_step(env, actions_dict)

                # SUMO only sets __all__=True; individual agent dones are always
                # False. Propagate the episode boundary to all active agents.
                episode_ended = isinstance(dones_dict, dict) and dones_dict.get(
                    "__all__", False
                )

                for ts in active_ts:
                    if isinstance(rewards_dict, dict) and ts in rewards_dict:
                        reward_value = float(rewards_dict[ts])
                        if not np.isfinite(reward_value):
                            reward_value = 0.0
                        reward_value = float(np.clip(reward_value, -1000.0, 1000.0))
                    else:
                        reward_value = 0.0
                    epoch_reward_sum[ts] += reward_value

                    scaled_reward = reward_value * REWARD_SCALE
                    if episode_ended:
                        # The episode ends by time limit (truncation), not a
                        # terminal state — fold the bootstrap value of the final
                        # observation into the last reward (SB3-style).
                        final_obs = (
                            next_obs_dict.get(ts)
                            if isinstance(next_obs_dict, dict)
                            else None
                        )
                        if final_obs is not None:
                            with torch.no_grad():
                                v_final = agents[ts].get_value(
                                    obs_to_tensor(final_obs, obs_dims[ts], device)
                                ).view(-1)[0]
                            scaled_reward += GAMMA * float(v_final)

                    memories[ts]["rewards"].append(scaled_reward)
                    memories[ts]["dones"].append(1.0 if episode_ended else 0.0)

                obs_dict = next_obs_dict
                if episode_ended:
                    obs_dict = env_reset(env)

            diag = {
                "ev": [],
                "actor_loss": [],
                "critic_loss": [],
                "entropy": [],
                "approx_kl": [],
                "clip_frac": [],
                "grad_norm": [],
            }
            any_updates = False
            for ts in ts_ids:
                T = len(memories[ts]["obs"])
                if T == 0:
                    continue
                any_updates = True

                obs_t = torch.stack(memories[ts]["obs"]).to(device)
                mask_t = torch.stack(memories[ts]["masks"]).to(device)
                act_t = torch.stack(memories[ts]["actions"]).to(device).long()
                logp_t = torch.stack(memories[ts]["logprobs"]).to(device)
                rew_t = torch.tensor(
                    memories[ts]["rewards"], dtype=torch.float32, device=device
                )
                val_t = torch.stack(memories[ts]["values"]).to(device)
                don_t = torch.tensor(
                    memories[ts]["dones"], dtype=torch.float32, device=device
                )

                # GAE advantage computation. don_t=1 cuts bootstrapping and
                # advantage propagation at the episode boundary (the truncation
                # bootstrap is already folded into the final reward).
                advantages = torch.zeros_like(rew_t)
                lastgaelam = 0.0
                for t in reversed(range(T)):
                    nextnonterminal = 1.0 - don_t[t]
                    if t == T - 1:
                        if don_t[t] > 0.5:
                            nextvalue = torch.zeros((), device=device)
                        else:
                            # Rollout cut mid-episode: bootstrap from current obs.
                            if ts in obs_dict:
                                next_obs = obs_to_tensor(
                                    obs_dict[ts], obs_dims[ts], device
                                )
                            else:
                                next_obs = memories[ts]["obs"][-1]
                            with torch.no_grad():
                                nextvalue = agents[ts].get_value(next_obs).view(-1)[0]
                    else:
                        nextvalue = val_t[t + 1]

                    delta = rew_t[t] + GAMMA * nextvalue * nextnonterminal - val_t[t]
                    lastgaelam = (
                        delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                    )
                    advantages[t] = lastgaelam

                returns = advantages + val_t
                advantages = torch.nan_to_num(
                    advantages, nan=0.0, posinf=1000.0, neginf=-1000.0
                )
                returns = torch.nan_to_num(
                    returns, nan=0.0, posinf=1000.0, neginf=-1000.0
                )

                var_ret = returns.var(unbiased=False)
                if float(var_ret) > 1e-8:
                    diag["ev"].append(
                        1.0 - float((returns - val_t).var(unbiased=False) / var_ret)
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
                        mb_mask = mask_t[idx]
                        mb_act = act_t[idx]
                        mb_logp_old = logp_t[idx]
                        mb_adv = adv_b[idx]
                        mb_ret = returns[idx]

                        _, newlogprob, entropy, newvalue = agents[
                            ts
                        ].get_action_and_value(
                            mb_obs,
                            action=mb_act,
                            action_mask=mb_mask,
                        )

                        logratio = newlogprob - mb_logp_old
                        ratio = logratio.exp()

                        with torch.no_grad():
                            diag["approx_kl"].append(
                                ((ratio - 1.0) - logratio).mean().item()
                            )
                            diag["clip_frac"].append(
                                ((ratio - 1.0).abs() > CLIP_FRAC).float().mean().item()
                            )

                        # Pessimistic clipped objective: max() of the negated
                        # terms (min() would optimize the optimistic bound and
                        # disable the trust region).
                        pg_loss1 = -mb_adv * ratio
                        pg_loss2 = -mb_adv * torch.clamp(
                            ratio, 1 - CLIP_FRAC, 1 + CLIP_FRAC
                        )
                        actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Unclipped value loss: returns are unnormalized, so a
                        # +-CLIP_FRAC value clip saturates and freezes the critic.
                        critic_loss = 0.5 * ((newvalue - mb_ret) ** 2).mean()
                        entropy_loss = entropy.mean()

                        loss = (
                            actor_loss
                            - current_entropy_coef * entropy_loss
                            + PPO_VALUE_COEF * critic_loss
                        )

                        optimizers[ts].zero_grad()
                        loss.backward()
                        total_norm = nn.utils.clip_grad_norm_(
                            agents[ts].parameters(), PPO_MAX_GRAD_NORM
                        )
                        optimizers[ts].step()

                        diag["actor_loss"].append(actor_loss.item())
                        diag["critic_loss"].append(critic_loss.item())
                        diag["entropy"].append(entropy_loss.item())
                        diag["grad_norm"].append(total_norm.item())

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

            def _mean_diag(key: str) -> float:
                return float(np.mean(diag[key])) if diag[key] else float("nan")

            log(
                "Diagnostyka: "
                f"explained_var={_mean_diag('ev'):.3f} | "
                f"actor_loss={_mean_diag('actor_loss'):.4f} | "
                f"critic_loss={_mean_diag('critic_loss'):.4f} | "
                f"entropia={_mean_diag('entropy'):.3f} | "
                f"approx_KL={_mean_diag('approx_kl'):.4f} | "
                f"clip_frac={_mean_diag('clip_frac'):.3f} | "
                f"grad_norm={_mean_diag('grad_norm'):.2f}"
            )

            if update % TRAIN_EVAL_EVERY_UPDATES == 0:
                eval_score, eval_metrics = evaluate_agent(
                    agents,
                    device,
                    ts_ids,
                    obs_dims,
                    TRAIN_EVAL_SEED,
                )
                log(
                    f"Ocena greedy: {eval_score:.3f} | "
                    f"sr. czekanie={eval_metrics['mean_waiting_time']:.1f}s | "
                    f"dojechalo={eval_metrics['arrived']:.0f} | "
                    f"teleporty={eval_metrics['teleported']:.0f} | "
                    f"backlog na koncu={eval_metrics['backlogged_end']:.0f}"
                )
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    torch.save(
                        _build_checkpoint(agents, ts_ids, obs_dims, act_dims, best_eval_score, update),
                        best_weights_file_path,
                    )
                    log(
                        f"Nowy najlepszy model: {best_weights_file_path} (score={best_eval_score:.3f})"
                    )

            last_completed_update = update
            if update % RESUME_SAVE_EVERY_UPDATES == 0:
                _save_resume_state(
                    resume_file_path,
                    agents,
                    optimizers,
                    ts_ids,
                    obs_dims,
                    act_dims,
                    update,
                    best_eval_score,
                )
                log(f"Zapisano punkt wznowienia (epoka {update}).")

        torch.save(
            _build_checkpoint(agents, ts_ids, obs_dims, act_dims, best_eval_score),
            weights_file_path,
        )
        log(f"\nTrening zakończony pomyślnie. Wagi zapisano w {weights_file_path}.")
        if os.path.exists(best_weights_file_path):
            log(f"Najlepszy checkpoint zapisano w {best_weights_file_path}")
        if os.path.exists(resume_file_path):
            os.remove(resume_file_path)
            log("Usunięto punkt wznowienia (trening ukończony).")
    except KeyboardInterrupt:
        if ready:
            _save_resume_state(
                resume_file_path,
                agents,
                optimizers,
                ts_ids,
                obs_dims,
                act_dims,
                last_completed_update,
                best_eval_score,
            )
            log(
                "\nPrzerwano przez użytkownika. Zapisano punkt wznowienia na epoce "
                f"{last_completed_update}."
            )
            log(f"Aby wznowić: python -m src.agent_ppo --resume {run_id}")
        else:
            log("\nPrzerwano przed rozpoczęciem treningu — nic do zapisania.")
    finally:
        if env is not None:
            env.close()
        log_file.close()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Trening niezależnego PPO dla mapy city_map_2."
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="RUN_ID",
        help="Wznów przerwany trening. Bez wartości wznawia najnowszy zapisany "
        "run; podaj numer (np. --resume 8), aby wskazać konkretny.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    resume_run_id = None
    if args.resume is not None:
        if args.resume == "auto":
            resume_run_id = get_latest_resume_run_id()
            if resume_run_id is None:
                raise SystemExit(
                    "Brak zapisanych punktów wznowienia w outputs/ — uruchom "
                    "trening normalnie (bez --resume)."
                )
        else:
            resume_run_id = int(args.resume)
        print(f"Wznawiam trening run {resume_run_id}...")
    else:
        print("Startuję niezależne PPO (oddzielne polityki) dla mapy city_map_2...")
    train(resume_run_id=resume_run_id)
