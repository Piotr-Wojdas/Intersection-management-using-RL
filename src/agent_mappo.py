"""MAPPO: decentralized actors + one centralized (global) critic.

Each intersection keeps its own actor that sees only its local observation, so
execution and evaluation are identical to IPPO (eval_model.py loads these
checkpoints unchanged). Training differs in two ways:

1. A single **centralized critic** estimates the value of the *global* state
   (the concatenation of every agent's observation), giving a lower-variance,
   coordination-aware baseline.
2. Agents optimise a **team reward** (the mean of the per-agent rewards). This
   internalises externalities: an agent that floods a downstream bottleneck
   (e.g. J2) now hurts the shared reward, so neighbours stop doing it. A global
   critic alone would only cut variance — the team reward is what changes the
   selfish incentive.

Run:    python -m src.agent_mappo
Resume: python -m src.agent_mappo --resume <run_id>
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import src.params as P  # scenario-dependent values read via P.X (see apply_scenario)
import src.setup_sumo  # noqa: F401  (sets SUMO_HOME before any TraCI/sumolib use)
from src.agent_ppo import (
    PPOAgent,
    _action_mask_tensor,
    _aggregate_policy_reward,
    _layer_init,
    _scheduled_entropy_coef,
    _scheduled_lr,
    _schedule_progress,
    _set_optimizer_lr,
    build_env,
    check_traffic_pool,
    train_reset,
)
from src.params import (
    CENTRAL_CRITIC_HIDDEN_DIMS,
    CLIP_FRAC,
    EPOCHS,
    GAMMA,
    GAE_LAMBDA,
    GLOBAL_SEED,
    LEARNING_RATE,
    MAPPO_BACKLOG_PENALTY_WEIGHT,
    NET_FILE,
    NUM_UPDATES,
    PPO_ENTROPY_COEF,
    PPO_ENTROPY_FINAL_FRAC,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH_SIZE,
    RESUME_SAVE_EVERY_UPDATES,
    ROLLOUT_STEPS,
    TRAIN_EVAL_EVERY_UPDATES,
    apply_scenario,
    build_resume_file,
    build_training_artifacts,
    get_latest_resume_run_id,
    resolve_eval_route_file,
    scenario_is_hard,
)
from src.utils import env_reset, env_step, make_log_fn, obs_to_tensor, resolve_device


class CentralizedCritic(nn.Module):
    """Value network over the global state (all agents' observations concatenated)."""

    def __init__(self, global_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = list(CENTRAL_CRITIC_HIDDEN_DIMS)
        layers = []
        last_dim = global_dim
        for hidden_dim in hidden_dims:
            layers.append(_layer_init(nn.Linear(last_dim, hidden_dim)))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.value = _layer_init(nn.Linear(last_dim, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.value(self.trunk(x)).squeeze(-1)


def _global_state(obs_dict, ts_ids, obs_dims, device) -> torch.Tensor:
    """Concatenate all agents' observations into the global state (fixed order)."""
    parts = []
    for ts in ts_ids:
        if ts in obs_dict:
            parts.append(obs_to_tensor(obs_dict[ts], obs_dims[ts], device))
        else:
            parts.append(torch.zeros(obs_dims[ts], dtype=torch.float32, device=device))
    return torch.cat(parts)


def _evaluate_mappo(actors, device, ts_ids, obs_dims, eval_seeds):
    """Greedy evaluation scored by the SAME team objective used for training
    (mean per-agent reward minus the global backlog penalty), averaged over the
    seeds. Selecting the best model by this score — rather than the raw
    congestion reward — stops a backlog-throttling policy from being saved as
    "best". Returns (score, averaged system metrics)."""
    if isinstance(eval_seeds, int):
        eval_seeds = [eval_seeds]
    scores, metrics_list = [], []
    for seed in eval_seeds:
        env = build_env(use_gui=False, sumo_seed=seed, route_file=resolve_eval_route_file())
        try:
            obs_dict = env_reset(env)
            team_sum = 0.0
            wait_means, last_info = [], {}
            done = False
            while not done:
                actions = {
                    ts: actors[ts].get_greedy_action(
                        obs_to_tensor(obs_dict[ts], obs_dims[ts], device),
                        action_mask=_action_mask_tensor(env, ts, device),
                    )
                    for ts in ts_ids
                    if ts in obs_dict
                }
                obs_dict, rewards_dict, dones_dict, info = env_step(env, actions)
                per = (
                    [float(rewards_dict.get(ts, 0.0)) for ts in ts_ids]
                    if isinstance(rewards_dict, dict)
                    else [0.0]
                )
                backlog = (
                    float(info.get("system_total_backlogged", 0.0))
                    if isinstance(info, dict)
                    else 0.0
                )
                team_sum += float(np.mean(per)) - MAPPO_BACKLOG_PENALTY_WEIGHT * backlog
                if isinstance(info, dict):
                    last_info = info
                    if "system_mean_waiting_time" in info:
                        wait_means.append(float(info["system_mean_waiting_time"]))
                done = (
                    bool(dones_dict.get("__all__", False))
                    if isinstance(dones_dict, dict)
                    else bool(dones_dict)
                )
            scores.append(team_sum)
            metrics_list.append(
                {
                    "mean_waiting_time": float(np.mean(wait_means)) if wait_means else 0.0,
                    "arrived": float(last_info.get("system_total_arrived", 0.0)),
                    "teleported": float(last_info.get("system_total_teleported", 0.0)),
                    "backlogged_end": float(last_info.get("system_total_backlogged", 0.0)),
                }
            )
        finally:
            env.close()
    avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
    return float(np.mean(scores)), avg_metrics


def _build_checkpoint(actors, critic, ts_ids, obs_dims, act_dims, best_eval_score, update=None):
    ckpt = {
        "algo": "mappo",
        "model_state_dict": {ts: actors[ts].state_dict() for ts in ts_ids},
        "central_critic_state_dict": critic.state_dict(),
        "obs_dims": obs_dims,
        "act_dims": act_dims,
        "ts_ids": ts_ids,
        "map_net_file": NET_FILE,
        "map_route_file": P.ROUTE_FILE,
        "best_eval_score": best_eval_score,
    }
    if update is not None:
        ckpt["update"] = update
    return ckpt


def _build_resume_state(
    actors, critic, actor_opts, critic_opt, ts_ids, obs_dims, act_dims, update, best_eval_score
):
    state = _build_checkpoint(actors, critic, ts_ids, obs_dims, act_dims, best_eval_score, update)
    state.update(
        {
            "actor_optimizer_state_dict": {ts: actor_opts[ts].state_dict() for ts in ts_ids},
            "critic_optimizer_state_dict": critic_opt.state_dict(),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
        }
    )
    return state


def _save_resume_state(path, *args):
    """Atomic write (tmp + os.replace) so an interrupt cannot corrupt the file."""
    tmp_path = f"{path}.tmp"
    torch.save(_build_resume_state(*args), tmp_path)
    os.replace(tmp_path, path)


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
                f"Brak pliku wznowienia dla run {resume_run_id}: {resume_file}."
            )
        resume_state = torch.load(resume_file, map_location=device, weights_only=False)
        if "central_critic_state_dict" not in resume_state:
            raise ValueError(
                f"Plik wznowienia run {resume_run_id} nie jest treningiem MAPPO "
                "(brak centralnego krytyka). Wznów go przez src.agent_ppo."
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
    actors: dict[str, PPOAgent] = {}
    actor_opts: dict[str, optim.Optimizer] = {}
    critic = None
    critic_opt = None
    try:
        if resume_state is not None:
            log(f"\n=== Wznowienie treningu MAPPO (run {run_id}) ===")
        else:
            log("=== Trening MAPPO (centralny krytyk) ===")
        log(f"Plik wag: {weights_file_path}")
        log(f"Plik najlepszego modelu: {best_weights_file_path}")
        log(f"Plik punktu wznowienia: {resume_file_path}")
        log(f"Plik logu: {log_file_path}")
        if P.RANDOMIZE_TRAFFIC:
            log(
                f"Domain randomization: pula {len(P.TRAFFIC_POOL_FILES)} plików treningowych, "
                f"held-out eval = {os.path.basename(P.EVAL_ROUTE_FILE)}"
            )
        else:
            log(f"Plik tras: {P.ROUTE_FILE}")
        log(f"Używam urządzenia: {device} | seed={GLOBAL_SEED}")

        check_traffic_pool()
        env = build_env(use_gui=False)
        ts_ids = env.ts_ids

        obs_dims = {ts: int(env.observation_spaces(ts).shape[0]) for ts in ts_ids}
        act_dims = {ts: int(env.action_spaces(ts).n) for ts in ts_ids}
        global_dim = sum(obs_dims[ts] for ts in ts_ids)

        for ts in ts_ids:
            actors[ts] = PPOAgent(obs_dims[ts], act_dims[ts]).to(device)
            actor_opts[ts] = optim.Adam(actors[ts].parameters(), lr=LEARNING_RATE)
        critic = CentralizedCritic(global_dim).to(device)
        critic_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

        log(
            f"Aktorzy: {len(ts_ids)} skrzyżowań (obs_dims={obs_dims}); "
            f"globalny krytyk: wejście {global_dim}."
        )

        best_eval_score = -float("inf")
        start_update = 1
        if resume_state is not None:
            if resume_state.get("obs_dims") != obs_dims or resume_state.get("act_dims") != act_dims:
                raise ValueError(
                    "Wymiary sieci w pliku wznowienia nie pasują do środowiska."
                )
            for ts in ts_ids:
                actors[ts].load_state_dict(resume_state["model_state_dict"][ts])
                actor_opts[ts].load_state_dict(resume_state["actor_optimizer_state_dict"][ts])
            critic.load_state_dict(resume_state["central_critic_state_dict"])
            critic_opt.load_state_dict(resume_state["critic_optimizer_state_dict"])
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
        obs_dict = train_reset(env)

        for update in range(start_update, NUM_UPDATES + 1):
            progress = _schedule_progress(update, NUM_UPDATES)
            current_lr = _scheduled_lr(progress)
            current_entropy_coef = _scheduled_entropy_coef(progress)
            for opt in actor_opts.values():
                _set_optimizer_lr(opt, current_lr)
            _set_optimizer_lr(critic_opt, current_lr)

            log(
                f"\nUruchamiam epokę uczenia #{update} | lr={current_lr:.6f} | "
                f"entropy_coef={current_entropy_coef:.6f}"
            )

            actor_mem = {
                ts: {"obs": [], "masks": [], "actions": [], "logprobs": []}
                for ts in ts_ids
            }
            gstates, cvalues, team_rewards, dones = [], [], [], []
            epoch_reward_sum = {ts: 0.0 for ts in ts_ids}

            for step in range(ROLLOUT_STEPS):
                if step == 0:
                    missing = [ts for ts in ts_ids if ts not in obs_dict]
                    if missing:
                        raise RuntimeError(
                            "MAPPO zakłada, że wszyscy agenci działają co krok; "
                            f"brakuje obserwacji dla {missing}."
                        )

                gstate = _global_state(obs_dict, ts_ids, obs_dims, device)
                with torch.no_grad():
                    cvalue = critic(gstate).view(-1)[0]

                actions_dict = {}
                for ts in ts_ids:
                    obs_ts = obs_to_tensor(obs_dict[ts], obs_dims[ts], device)
                    mask_ts = _action_mask_tensor(env, ts, device)
                    with torch.no_grad():
                        action, logprob, _, _ = actors[ts].get_action_and_value(
                            obs_ts, action_mask=mask_ts
                        )
                    actions_dict[ts] = int(action.item())
                    mem = actor_mem[ts]
                    mem["obs"].append(obs_ts)
                    mem["masks"].append(mask_ts)
                    mem["actions"].append(action.view(()))
                    mem["logprobs"].append(logprob.view(()))

                next_obs_dict, rewards_dict, dones_dict, info = env_step(env, actions_dict)
                episode_ended = isinstance(dones_dict, dict) and dones_dict.get(
                    "__all__", False
                )
                backlog = (
                    float(info.get("system_total_backlogged", 0.0))
                    if isinstance(info, dict)
                    else 0.0
                )

                per_agent_r = []
                for ts in ts_ids:
                    if isinstance(rewards_dict, dict) and ts in rewards_dict:
                        r = float(rewards_dict[ts])
                        if not np.isfinite(r):
                            r = 0.0
                        r = float(np.clip(r, -1000.0, 1000.0))
                    else:
                        r = 0.0
                    epoch_reward_sum[ts] += r
                    per_agent_r.append(r)

                # Team reward = mean per-agent congestion reward MINUS a global
                # backlog penalty. Backlogged (pending) vehicles never touch a
                # controlled lane, so they are invisible to the congestion
                # reward — without this term the policy can raise its reward by
                # throttling entry and stranding cars at the gate. A global
                # penalty is only coherent with a shared objective, which the
                # centralized critic provides.
                raw_team = float(np.mean(per_agent_r)) - MAPPO_BACKLOG_PENALTY_WEIGHT * backlog
                team_reward = raw_team * P.REWARD_SCALE
                if episode_ended:
                    gnext = _global_state(next_obs_dict, ts_ids, obs_dims, device)
                    with torch.no_grad():
                        team_reward += GAMMA * float(critic(gnext).view(-1)[0])

                gstates.append(gstate)
                cvalues.append(cvalue)
                team_rewards.append(team_reward)
                dones.append(1.0 if episode_ended else 0.0)

                obs_dict = next_obs_dict
                if episode_ended:
                    obs_dict = train_reset(env)

            # --- Centralized GAE on the team reward ---
            T = len(gstates)
            gs_t = torch.stack(gstates).to(device)
            cval_t = torch.stack(cvalues).to(device).view(-1)
            rew_t = torch.tensor(team_rewards, dtype=torch.float32, device=device)
            don_t = torch.tensor(dones, dtype=torch.float32, device=device)

            advantages = torch.zeros_like(rew_t)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                nextnonterminal = 1.0 - don_t[t]
                if t == T - 1:
                    if don_t[t] > 0.5:
                        nextvalue = torch.zeros((), device=device)
                    else:
                        gnext = _global_state(obs_dict, ts_ids, obs_dims, device)
                        with torch.no_grad():
                            nextvalue = critic(gnext).view(-1)[0]
                else:
                    nextvalue = cval_t[t + 1]
                delta = rew_t[t] + GAMMA * nextvalue * nextnonterminal - cval_t[t]
                lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam

            returns = advantages + cval_t
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1000.0, neginf=-1000.0)
            returns = torch.nan_to_num(returns, nan=0.0, posinf=1000.0, neginf=-1000.0)

            var_ret = returns.var(unbiased=False)
            explained_var = (
                float(1.0 - (returns - cval_t).var(unbiased=False) / var_ret)
                if float(var_ret) > 1e-8
                else float("nan")
            )
            # Shared, standardized advantage drives every actor — this is the
            # coordination signal: each agent is pushed by the team's outcome.
            adv_b = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            obs_t = {ts: torch.stack(actor_mem[ts]["obs"]).to(device) for ts in ts_ids}
            mask_t = {ts: torch.stack(actor_mem[ts]["masks"]).to(device) for ts in ts_ids}
            act_t = {ts: torch.stack(actor_mem[ts]["actions"]).to(device).long() for ts in ts_ids}
            logp_t = {ts: torch.stack(actor_mem[ts]["logprobs"]).to(device) for ts in ts_ids}

            diag = {"actor_loss": [], "critic_loss": [], "entropy": [],
                    "approx_kl": [], "clip_frac": [], "actor_gn": [], "critic_gn": []}

            minibatch_size = min(PPO_MINIBATCH_SIZE, T)
            for _epoch in range(EPOCHS):
                permutation = torch.randperm(T, device=device)
                for start in range(0, T, minibatch_size):
                    idx = permutation[start : start + minibatch_size]
                    mb_adv = adv_b[idx]

                    # Centralized critic update (separate optimizer; unclipped MSE).
                    new_value = critic(gs_t[idx])
                    critic_loss = 0.5 * ((new_value - returns[idx]) ** 2).mean()
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_gn = nn.utils.clip_grad_norm_(critic.parameters(), PPO_MAX_GRAD_NORM)
                    critic_opt.step()
                    diag["critic_loss"].append(critic_loss.item())
                    diag["critic_gn"].append(critic_gn.item())

                    # Each actor: clipped PPO objective with the shared advantage.
                    for ts in ts_ids:
                        _, newlogprob, entropy, _ = actors[ts].get_action_and_value(
                            obs_t[ts][idx], action=act_t[ts][idx], action_mask=mask_t[ts][idx]
                        )
                        logratio = newlogprob - logp_t[ts][idx]
                        ratio = logratio.exp()
                        pg_loss1 = -mb_adv * ratio
                        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_FRAC, 1 + CLIP_FRAC)
                        actor_loss = torch.max(pg_loss1, pg_loss2).mean()
                        entropy_loss = entropy.mean()
                        loss = actor_loss - current_entropy_coef * entropy_loss

                        actor_opts[ts].zero_grad()
                        loss.backward()
                        actor_gn = nn.utils.clip_grad_norm_(
                            actors[ts].parameters(), PPO_MAX_GRAD_NORM
                        )
                        actor_opts[ts].step()

                        with torch.no_grad():
                            diag["approx_kl"].append(((ratio - 1.0) - logratio).mean().item())
                            diag["clip_frac"].append(
                                ((ratio - 1.0).abs() > CLIP_FRAC).float().mean().item()
                            )
                        diag["actor_loss"].append(actor_loss.item())
                        diag["entropy"].append(entropy_loss.item())
                        diag["actor_gn"].append(actor_gn.item())

            per_agent_summary = ", ".join(f"{ts}={epoch_reward_sum[ts]:.3f}" for ts in ts_ids)
            log(f"Nagroda per agent: {per_agent_summary}")
            log(f"Srednia nagroda na agenta (zespolowa): {_aggregate_policy_reward(epoch_reward_sum):.3f}")

            def _m(key):
                return float(np.mean(diag[key])) if diag[key] else float("nan")

            log(
                "Diagnostyka: "
                f"explained_var={explained_var:.3f} | "
                f"actor_loss={_m('actor_loss'):.4f} | "
                f"critic_loss={_m('critic_loss'):.4f} | "
                f"entropia={_m('entropy'):.3f} | "
                f"approx_KL={_m('approx_kl'):.4f} | "
                f"clip_frac={_m('clip_frac'):.3f} | "
                f"grad_norm(aktor/krytyk)={_m('actor_gn'):.2f}/{_m('critic_gn'):.2f}"
            )

            if update % TRAIN_EVAL_EVERY_UPDATES == 0:
                eval_score, eval_metrics = _evaluate_mappo(
                    actors, device, ts_ids, obs_dims, P.TRAIN_EVAL_SEED
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
                        _build_checkpoint(actors, critic, ts_ids, obs_dims, act_dims, best_eval_score, update),
                        best_weights_file_path,
                    )
                    log(f"Nowy najlepszy model: {best_weights_file_path} (score={best_eval_score:.3f})")

            last_completed_update = update
            if update % RESUME_SAVE_EVERY_UPDATES == 0:
                _save_resume_state(
                    resume_file_path, actors, critic, actor_opts, critic_opt,
                    ts_ids, obs_dims, act_dims, update, best_eval_score,
                )
                log(f"Zapisano punkt wznowienia (epoka {update}).")

        torch.save(
            _build_checkpoint(actors, critic, ts_ids, obs_dims, act_dims, best_eval_score),
            weights_file_path,
        )
        log(f"\nTrening MAPPO zakończony. Wagi zapisano w {weights_file_path}.")
        if os.path.exists(best_weights_file_path):
            log(f"Najlepszy checkpoint zapisano w {best_weights_file_path}")
        if os.path.exists(resume_file_path):
            os.remove(resume_file_path)
            log("Usunięto punkt wznowienia (trening ukończony).")
    except KeyboardInterrupt:
        if ready:
            _save_resume_state(
                resume_file_path, actors, critic, actor_opts, critic_opt,
                ts_ids, obs_dims, act_dims, last_completed_update, best_eval_score,
            )
            log(
                "\nPrzerwano przez użytkownika. Zapisano punkt wznowienia na epoce "
                f"{last_completed_update}."
            )
            log(f"Aby wznowić: python -m src.agent_mappo --resume {run_id}")
        else:
            log("\nPrzerwano przed rozpoczęciem treningu — nic do zapisania.")
    finally:
        if env is not None:
            env.close()
        log_file.close()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Trening MAPPO (centralny krytyk) dla mapy city_map_2."
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="RUN_ID",
        help="Wznów przerwany trening MAPPO. Bez wartości wznawia najnowszy run.",
    )
    parser.add_argument(
        "--scenario",
        choices=["easy", "hard"],
        default=None,
        help="Trudność ruchu. Domyślnie bierze USE_HARD_TRAFFIC z params.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    apply_scenario(scenario_is_hard(args.scenario))
    resume_run_id = None
    if args.resume is not None:
        if args.resume == "auto":
            resume_run_id = get_latest_resume_run_id()
            if resume_run_id is None:
                raise SystemExit("Brak zapisanych punktów wznowienia w outputs/.")
        else:
            resume_run_id = int(args.resume)
        print(f"Wznawiam trening MAPPO run {resume_run_id}...")
    else:
        print("Startuję MAPPO (centralny krytyk) dla mapy city_map_2...")
    train(resume_run_id=resume_run_id)
