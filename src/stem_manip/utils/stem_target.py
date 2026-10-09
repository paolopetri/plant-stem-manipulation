"""Target positions for the stem point (stage 1): on or below the "bowl" it reaches when pushed, within a curvature
budget (user, 2026-10-08/09).

The stem does not stretch, so a point at arc length s pushed sideways by r also drops. Shape model: a clamped stem
with a force and a moment at the point, w(x) = a x^2 + b x^3 (small deflection). With the tip slope t (in units of
r / s) the shape is fixed and
- drop = c r^2 / s with c(t) = t^2 / 15 - t / 10 + 3 / 5 (`drop_factor`; 1/2 int w'^2);
- peak curvature = g(t) r / s^2 with g(t) = max(|6 - 2 t|, |4 t - 6|), at the base or at the point
  (`curvature_factor`).
A force alone (t = 1.5) gives the "bowl", c = 0.6, g = 3. Deeper targets (c > 0.6) need C / S shapes, i.e. a moment
from the fork's slot; they bend the stem more, up to the budget. Above the bowl the stem reaches at most 0.2-1.7 mm,
and only with its base bent to the limit (review 2026-10-09), so no target is placed there.
The shape depends on the stem's length and curvature limit, not on its stiffness (stiffness sets the force needed).

Assumes an upright stem in a frame with z up (robot base frame). Batched over envs, no simulator needed.
Tests: `tests/test_stem_target.py`.
"""

import torch

BOWL_FACTOR = 0.6
"""Drop factor c of the stem pushed at the point by a force alone (tip slope 1.5 r / s)."""


def bowl_drop(distance: torch.Tensor, arc_length: float) -> torch.Tensor:
    """Drop [m] of a point at `arc_length` [m] on an upright cantilever pushed sideways by `distance` [m] at it."""
    return BOWL_FACTOR * distance**2 / arc_length


def drop_factor(tip_slope: torch.Tensor) -> torch.Tensor:
    """Drop factor c (drop = c r^2 / s) of the shape with tip slope `tip_slope` [r / s]."""
    return tip_slope**2 / 15.0 - tip_slope / 10.0 + 0.6


def curvature_factor(tip_slope: torch.Tensor) -> torch.Tensor:
    """Peak-curvature factor g (curvature = g r / s^2) of the shape with tip slope `tip_slope` [r / s]."""
    return torch.maximum((6.0 - 2.0 * tip_slope).abs(), (4.0 * tip_slope - 6.0).abs())


def gentlest_tip_slope(factor: torch.Tensor) -> torch.Tensor:
    """Tip slope [r / s] of the gentlest shape with drop factor `factor` >= 0.6 (the larger root t of
    c(t) = factor; the smaller root 1.5 - t is <= 0 and bends the stem more: g = 4 t)."""
    return 0.75 + torch.sqrt(15.0 * factor - 8.4375)


def deepest_drop_factor(distance: torch.Tensor, arc_length: float, max_curvature: float) -> torch.Tensor:
    """Largest drop factor c whose gentlest shape stays within `max_curvature` [1/m] at sideways `distance` [m]:
    g(t) = 4 t - 6 = max_curvature s^2 / r. Not below the bowl's 0.6 (the bowl itself needs g = 3)."""
    tip_slope = (max_curvature * arc_length**2 / distance + 6.0) / 4.0
    return drop_factor(tip_slope).clamp(min=BOWL_FACTOR)


def sample_targets(
    rest_points: torch.Tensor,
    arc_length: float,
    distance_range: tuple[float, float],
    angle_range: tuple[float, float],
    max_curvature: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Target positions around the rest positions of the stem point, shape `(num_envs, 3)` [m].

    Args:
        rest_points: Rest position of the stem point per env, shape `(num_envs, 3)` [m], z up.
        arc_length: Arc length of the stem point from the base [m].
        distance_range: Range of the horizontal distance r from the rest position [m], sampled uniformly.
        angle_range: Range of the horizontal direction [rad] (angle from +x towards +y), sampled uniformly.
        max_curvature: Curvature budget of the deepest target shape [1/m]; the drop factor is sampled uniformly
            between the bowl (0.6) and `deepest_drop_factor`.
        generator: Random generator on the device of `rest_points` (reproducible tests); `None` = torch's default.

    Returns:
        The targets, in the frame of `rest_points`.
    """
    num_envs, device = rest_points.shape[0], rest_points.device

    def uniform(low: float | torch.Tensor, high: float | torch.Tensor) -> torch.Tensor:
        u = torch.rand(num_envs, generator=generator, device=device, dtype=rest_points.dtype)
        return low + (high - low) * u

    distance = uniform(*distance_range)
    angle = uniform(*angle_range)
    factor = uniform(BOWL_FACTOR, deepest_drop_factor(distance, arc_length, max_curvature))
    offset = torch.stack(
        (distance * torch.cos(angle), distance * torch.sin(angle), -factor * distance**2 / arc_length), dim=-1
    )
    return rest_points + offset
