"""Shared utilities used by training and evaluation scripts."""

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
