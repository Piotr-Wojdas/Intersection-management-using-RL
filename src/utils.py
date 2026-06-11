"""Shared utilities used by training and evaluation scripts."""

from typing import IO

import torch


def resolve_device() -> torch.device:
    from src.params import DEVICE_OVERRIDE, USE_CUDA_IF_AVAILABLE

    if DEVICE_OVERRIDE is not None:
        return torch.device(DEVICE_OVERRIDE)
    if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def obs_to_tensor(obs, expected_dim: int, device) -> torch.Tensor:
    """Convert an observation to a flat float32 tensor and validate its size.

    A size mismatch means the observation function changed relative to the
    network/checkpoint — fail loudly instead of silently padding with zeros.
    """
    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).view(-1)
    if tensor.shape[0] != expected_dim:
        raise ValueError(
            f"Observation has dim {tensor.shape[0]}, expected {expected_dim}."
        )
    return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)


def env_reset(env) -> dict:
    """Reset env and unwrap the (obs, info) tuple that Gymnasium returns."""
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def env_step(env, actions: dict) -> tuple[dict, dict, dict, dict]:
    """Step env and normalize the 4- vs 5-element return to (obs, rewards, dones, info).

    Old SUMO-RL returns (obs, rewards, dones, info).
    Gymnasium returns   (obs, rewards, terminated, truncated, info).
    Both are collapsed to a unified 4-tuple so callers don't need the branching.
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
