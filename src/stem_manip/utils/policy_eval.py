"""Summary of a policy evaluation (no simulator imports): how the episodes end, the final errors, the fork rotation.

`scripts/evaluate_policy.py` collects the first episode of every env into `Episodes` and prints `summarize`.
Tests: `tests/test_policy_eval.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from stem_manip.utils.quat import quat_conjugate, quat_mul, rotvec_from_quat


@dataclass
class Episodes:
    """Per-episode values of n finished episodes, each a tensor (n,)."""

    ended_by: dict[str, torch.Tensor]
    """Termination term name -> whether it fired at the episode's last step (bool); several terms can fire at once."""
    length_s: torch.Tensor
    """Episode length [s]."""
    position_error: torch.Tensor
    """Distance of the stem point to its target at the end [m] (the command metric of the last step before it)."""
    height_error: torch.Tensor
    """z part of that error at the end [m], + = above the target."""
    max_rotation: torch.Tensor
    """Largest rotation angle of the slot base from its orientation at the reset during the episode [rad]."""
    final_rotation: torch.Tensor
    """Rotation angle of the slot base from its orientation at the reset at the end [rad]."""


def summarize(episodes: Episodes, time_out: str, success_distance: float) -> dict[str, float]:
    """Summary of the episodes, key -> value.

    Args:
        episodes: The evaluated episodes.
        time_out: Name of the time-out term; every other term is an early termination.
        success_distance: Final position error that counts as reached [m].

    Returns:
        `episodes`, `length_s/mean`, `ended_by/<term>` and `ended_by/early (any)` (shares of the episodes);
        `<group>/position_error/...`, `<group>/reached` (share under `success_distance`) and
        `<group>/height_error/...` for the groups `all` and `time_out` (left out without time-out episodes);
        `all/max_rotation/...` and `all/final_rotation/...`.
    """
    summary = {"episodes": len(episodes.length_s), "length_s/mean": episodes.length_s.mean().item()}
    for name, fired in episodes.ended_by.items():
        summary[f"ended_by/{name}"] = fired.float().mean().item()
    early = torch.stack([fired for name, fired in episodes.ended_by.items() if name != time_out]).any(dim=0)
    summary["ended_by/early (any)"] = early.float().mean().item()
    for group, mask in (("all", torch.ones_like(early)), ("time_out", episodes.ended_by[time_out])):
        if not mask.any():
            continue
        position_error, height_error = episodes.position_error[mask], episodes.height_error[mask]
        summary.update(_stats(f"{group}/position_error", position_error, (0.5, 0.9)))
        summary[f"{group}/reached"] = (position_error < success_distance).float().mean().item()
        summary.update(_stats(f"{group}/height_error", height_error, (0.1, 0.5, 0.9)))
    summary.update(_stats("all/max_rotation", episodes.max_rotation, (0.5, 0.9)))
    summary.update(_stats("all/final_rotation", episodes.final_rotation, (0.5, 0.9)))
    return summary


def rotation_angle(quat_from: torch.Tensor, quat_to: torch.Tensor) -> torch.Tensor:
    """Angle (...,) [rad] in [0, pi] of the rotation from `quat_from` to `quat_to`, unit quaternions (..., 4)
    (x, y, z, w) in the same frame."""
    return rotvec_from_quat(quat_mul(quat_to, quat_conjugate(quat_from))).norm(dim=-1)


def _stats(name: str, values: torch.Tensor, quantiles: tuple[float, ...]) -> dict[str, float]:
    """`name/mean` and `name/median` or `name/p<q>` per quantile (linear interpolation, as numpy's default)."""
    stats = {f"{name}/mean": values.mean().item()}
    for q in quantiles:
        stats[f"{name}/median" if q == 0.5 else f"{name}/p{round(100 * q)}"] = torch.quantile(values, q).item()
    return stats
