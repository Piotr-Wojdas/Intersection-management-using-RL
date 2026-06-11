"""Shared utilities used by training and evaluation scripts."""

import io
from typing import IO

import torch


def resolve_device() -> torch.device:
    from src.params import DEVICE_OVERRIDE, USE_CUDA_IF_AVAILABLE

    if DEVICE_OVERRIDE is not None:
        return torch.device(DEVICE_OVERRIDE)
    if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pad_observation(obs: torch.Tensor, target_dim: int) -> torch.Tensor:
    if obs.dim() != 1:
        obs = obs.view(-1)
    if obs.shape[0] == target_dim:
        return obs
    padded = torch.zeros(target_dim, dtype=obs.dtype, device=obs.device)
    padded[: obs.shape[0]] = obs
    return padded


def env_reset(env) -> dict:
    """Reset env and unwrap the (obs, info) tuple that Gymnasium returns."""
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def env_step(env, actions: dict) -> tuple[dict, dict, dict]:
    """Step env and normalize the 4- vs 5-element return to (obs, rewards, dones).

    Old SUMO-RL returns (obs, rewards, dones, info).
    Gymnasium returns   (obs, rewards, terminated, truncated, info).
    Both are collapsed to a unified 3-tuple so callers don't need the branching.
    """
    res = env.step(actions)
    if len(res) == 5:
        obs, rewards, terminated, truncated, info = res
        dones = {"__all__": bool(terminated or truncated)}
    else:
        obs, rewards, dones, info = res
    return obs, rewards, dones, info


def make_log_fn(log_file: IO[str]):
    """Return a log function that writes to stdout and a file simultaneously."""

    def log(message: str = "") -> None:
        print(message)
        print(message, file=log_file, flush=True)

    return log
