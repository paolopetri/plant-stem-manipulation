"""Helpers of the scripted rollouts in the checks (no simulator imports): observation-term slices, zero / random
agent actions, a translation action toward a goal."""

import math
from collections.abc import Sequence

import torch


def obs_term_slices(names: Sequence[str], dims: Sequence[Sequence[int]]) -> dict[str, slice]:
    """Slice of each term in the concatenated observation group, in order (multi-dim terms flattened).

    Args:
        names: Term names, e.g. `env.observation_manager.active_terms["policy"]`.
        dims: Term shapes, e.g. `env.observation_manager.group_obs_term_dim["policy"]`.
    """
    slices, start = {}, 0
    for name, dim in zip(names, dims, strict=True):
        slices[name] = slice(start, start + math.prod(dim))
        start += math.prod(dim)
    return slices


def agent_action(agent: str, num_envs: int, action_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Action (num_envs, action_dim) of Isaac Lab's zero agent ("zero": all 0) or random agent ("random": uniform in
    [-1, 1], drawn from `generator`, on its device)."""
    if agent == "zero":
        return torch.zeros(num_envs, action_dim, device=generator.device)
    return 2 * torch.rand(num_envs, action_dim, device=generator.device, generator=generator) - 1


def toward(
    slot_base_pos: torch.Tensor, goal: torch.Tensor, max_step: float, max_step_change: float, speed: float = 1.0
) -> torch.Tensor:
    """Slot-base action (n, 6) that moves the slot base from `slot_base_pos` toward `goal` (n, 3) at most `speed` x
    the step cap `max_step`, and brakes in time for the step-change limit `max_step_change` (stopping distance
    s^2 / (2 ds) at step s); rotation part 0. The limits are those of `SlotBaseImpedanceActionCfg`."""
    offset = goal - slot_base_pos
    distance = offset.norm(dim=-1, keepdim=True)
    size = torch.sqrt(2.0 * max_step_change * distance).clamp(max=speed * max_step)
    action = torch.zeros(goal.shape[0], 6, dtype=goal.dtype, device=goal.device)
    action[:, :3] = offset / distance.clamp(min=1e-9) * size / max_step
    return action
