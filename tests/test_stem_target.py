"""Unit tests for `stem_manip.utils.stem_target`: target sampling for the stem tip (no simulator needed).

Requirements (user, 2026-10-08/09):
- Horizontal distance r of the target from the tip's rest position uniform in [0.03, 0.10] m, direction uniform in
  the angle range (default the full circle).
- Height: on or below the "bowl" (drop 0.6 r^2 / s: the stem pushed at its tip), down to the deepest shape whose
  peak curvature stays within the budget 0.8 x `damage.max_curvature` (C / S shapes need a moment from the fork's
  slot). Nothing above the bowl (the inextensible stem reaches at most 0.2-1.7 mm above it, with its base bent to
  the limit; review 2026-10-09).
- Shape model: a clamped stem with a force and a moment at the tip, w(x) = a x^2 + b x^3, small deflection; the
  closed-form drop and peak curvature match a numerical integration of that shape.
- Sampled per env relative to that env's rest position; different targets per env and per call.
"""

import math

import pytest
import torch

from stem_manip.assets.stem import stem_params
from stem_manip.utils import stem_target

ARC_LENGTH = 0.40  # [m] the tip (user, 2026-10-08)
DISTANCE_RANGE = (0.03, 0.10)  # [m] (user, 2026-10-08)
CURVATURE_BUDGET = 0.8  # fraction of damage.max_curvature for the deepest target shape (user, 2026-10-09)
MAX_CURVATURE = CURVATURE_BUDGET * stem_params("chain")["damage"]["max_curvature"]  # [1/m] 4 at 5 1/m
FULL_CIRCLE = (-math.pi, math.pi)
NUM_ENVS = 4096


def _rest_points() -> torch.Tensor:
    """Rest positions of the tip, different per env (as after a randomized spawn), shape (NUM_ENVS, 3)."""
    generator = torch.Generator().manual_seed(1)
    rest = torch.zeros(NUM_ENVS, 3, dtype=torch.float64)
    rest[:, 0] = 0.275 + 0.25 * torch.rand(NUM_ENVS, generator=generator, dtype=torch.float64)
    rest[:, 1] = -0.15 + 0.30 * torch.rand(NUM_ENVS, generator=generator, dtype=torch.float64)
    rest[:, 2] = ARC_LENGTH
    return rest


def _sample(angle_range: tuple[float, float] = FULL_CIRCLE, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rest = _rest_points()
    targets = stem_target.sample_targets(
        rest,
        ARC_LENGTH,
        DISTANCE_RANGE,
        angle_range,
        MAX_CURVATURE,
        generator=torch.Generator().manual_seed(seed),
    )
    return rest, targets


def _drop_factor(rest: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal distance r and drop factor c = drop * s / r^2 of each target."""
    distance = (targets[:, :2] - rest[:, :2]).norm(dim=-1)
    return distance, (rest[:, 2] - targets[:, 2]) * ARC_LENGTH / distance**2


@pytest.mark.parametrize("tip_slope", [-1.0, 0.0, 0.75, 1.5, 2.0, 3.1, 6.8])
def test_shape_factors_match_numerical_integration(tip_slope: float):
    """w(x) = a x^2 + b x^3 on s = 1 with w(1) = 1, w'(1) = tip_slope: drop 1/2 int w'^2, peak |w''|."""
    a, b = 3.0 - tip_slope, tip_slope - 2.0
    x = torch.linspace(0.0, 1.0, 20001, dtype=torch.float64)
    drop = 0.5 * torch.trapezoid((2 * a * x + 3 * b * x**2) ** 2, x)
    curvature = (2 * a + 6 * b * x).abs().max()
    slope = torch.tensor(tip_slope, dtype=torch.float64)
    assert float(stem_target.drop_factor(slope)) == pytest.approx(float(drop), rel=1e-6)
    assert float(stem_target.curvature_factor(slope)) == pytest.approx(float(curvature), rel=1e-9)


def test_tip_push_is_the_bowl():
    """Force at the tip only (tip slope 1.5 r/s): drop 0.6 r^2/s, peak curvature 3 r/s^2 at the base."""
    slope = torch.tensor(1.5, dtype=torch.float64)
    assert float(stem_target.drop_factor(slope)) == pytest.approx(0.6)
    assert float(stem_target.curvature_factor(slope)) == pytest.approx(3.0)
    assert float(stem_target.bowl_drop(torch.tensor(0.10), ARC_LENGTH)) == pytest.approx(0.015, rel=1e-6)


def test_deepest_shape_at_the_decided_budget():
    """At r = 10 cm the deepest shape within 4 1/m has c = 0.9307 (8.3 mm below the bowl), at 3 cm c = 3.0296."""
    deepest = stem_target.deepest_drop_factor(torch.tensor([0.10, 0.03], dtype=torch.float64), ARC_LENGTH, 4.0)
    torch.testing.assert_close(deepest, torch.tensor([0.9307, 3.0296], dtype=torch.float64), rtol=0.0, atol=1e-4)


def test_targets_lie_in_the_region():
    rest, targets = _sample()
    assert targets.shape == (NUM_ENVS, 3)
    distance, c = _drop_factor(rest, targets)
    assert float(distance.min()) >= DISTANCE_RANGE[0] - 1e-9
    assert float(distance.max()) <= DISTANCE_RANGE[1] + 1e-9
    deepest = stem_target.deepest_drop_factor(distance, ARC_LENGTH, MAX_CURVATURE)
    assert float(c.min()) >= 0.6 - 1e-9  # on or below the bowl
    assert bool((c <= deepest + 1e-9).all())
    # the whole band is used: targets near the bowl and near the deepest shape
    position = (c - 0.6) / (deepest - 0.6)
    assert float(position.min()) < 0.05 and float(position.max()) > 0.95


def test_targets_stay_within_the_curvature_budget():
    """The gentlest shape that reaches each target bends the stem by at most the budget."""
    rest, targets = _sample()
    distance, c = _drop_factor(rest, targets)
    slope = stem_target.gentlest_tip_slope(c)
    curvature = stem_target.curvature_factor(slope) * distance / ARC_LENGTH**2
    assert float(curvature.max()) <= MAX_CURVATURE + 1e-6
    assert float(stem_target.drop_factor(slope).sub(c).abs().max()) < 1e-9


def test_directions_cover_the_angle_range():
    rest, targets = _sample()
    angle = torch.atan2(targets[:, 1] - rest[:, 1], targets[:, 0] - rest[:, 0])
    counts = torch.histc(angle, bins=8, min=-math.pi, max=math.pi)
    assert bool((counts > 0.8 * NUM_ENVS / 8).all()), counts  # about uniform over the full circle

    quarter = (0.0, 0.5 * math.pi)
    rest, targets = _sample(angle_range=quarter)
    angle = torch.atan2(targets[:, 1] - rest[:, 1], targets[:, 0] - rest[:, 0])
    assert float(angle.min()) >= quarter[0] - 1e-6 and float(angle.max()) <= quarter[1] + 1e-6


def test_targets_differ_per_env_and_per_call():
    rest, first = _sample(seed=0)
    _, second = _sample(seed=1)
    offsets = first - rest
    assert torch.unique(offsets, dim=0).shape[0] == NUM_ENVS
    assert not torch.allclose(first, second)
