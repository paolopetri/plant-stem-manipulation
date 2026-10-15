"""Unit tests for `stem_manip.utils.stem_target`: target sampling for the stem tip (no simulator needed).
The tests check (decided values: user, 2026-10-08/09 and spec #160):

- Horizontal distance r of the target from the tip's rest position uniform in [0.03, 0.10] m, direction uniform in
  the angle range (default the full circle).
- Height: on or below the "bowl" (the drop of the stem pushed sideways at its tip, by a horizontal force alone),
  down to the deepest shape whose curvature stays within the budget 0.8 x `damage.max_curvature` (a moment or a C / S
  bend, or pressing the tip down). Nothing above the bowl.
- Shape model: a planar elastica (inextensible rod, clamped upright, a force and a moment at the stem point). The
  rod table matches an independent numerical integration (scipy's `solve_ivp`, shooting from the base): the bowl
  by a root search on the sideways force, the deepest shape at the parameters a constrained optimizer (SLSQP, scipy)
  found on 2026-10-15 (dimensionless, s = 1, EI = 1, budget 1.6).
- The region is dimensionless (r / s, drop / s, budget x s): the targets scale with s.
- Sampled per env relative to that env's rest position; different targets per env and per call.
"""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from stem_manip.utils import stem_target

ARC_LENGTH = 0.40  # [m] the tip (user, 2026-10-08)
DISTANCE_RANGE = (0.03, 0.10)  # [m] (user, 2026-10-08)
CURVATURE_AT_BUDGET = 4.0  # [1/m] the budget 0.8 (user, 2026-10-09) x damage.max_curvature 5 1/m
BUDGET = CURVATURE_AT_BUDGET * ARC_LENGTH  # dimensionless budget x s
FULL_CIRCLE = (-math.pi, math.pi)
NUM_ENVS = 4096

# deepest shapes within the budget 1.6 (dimensionless: base curvature k0, force fx, fz with fz > 0 pressing down),
# found by SLSQP over (k0, fx, fz) on 2026-10-15; keyed by r [m] at s = 0.40 m
DEEPEST_SHAPES = {
    0.03: (-1.37499003, 3.14540674, 14.60703736),
    0.05: (-1.24609713, 3.66818623, 13.10409426),
    0.075: (-1.05092303, 4.10885451, 11.23328683),
    0.10: (-0.82182576, 4.32260469, 9.44244472),
}


def _shoot(k0: float, fx: float, fz: float) -> np.ndarray:
    """Elastica from the clamped base, s = 1: theta' = kappa, kappa' = fx cos(theta) - fz sin(theta), (x, z) along
    (sin, cos). Rows theta, kappa, x, z over 2001 points."""

    def rhs(_, y):
        return [y[1], fx * np.cos(y[0]) - fz * np.sin(y[0]), np.sin(y[0]), np.cos(y[0])]

    sol = solve_ivp(rhs, (0.0, 1.0), [0.0, k0, 0.0, 0.0], rtol=1e-11, atol=1e-12, t_eval=np.linspace(0, 1, 2001))
    return sol.y


def _bowl(distance: float) -> float:
    """Drop [m] of the stem pushed sideways at its tip to `distance` [m]: horizontal force, no tip moment."""

    def base_curvature(fx: float) -> float:  # zero tip moment
        return brentq(lambda k0: _shoot(k0, fx, 0.0)[1, -1], 0.0, 2 * abs(fx) + 1e-3)

    fx = brentq(lambda f: _shoot(base_curvature(f), f, 0.0)[2, -1] - distance / ARC_LENGTH, -1e-6, -3.0)
    return (1.0 - _shoot(base_curvature(fx), fx, 0.0)[3, -1]) * ARC_LENGTH


def _rest_points() -> torch.Tensor:
    """Rest positions of the tip, different per env (as after a randomized spawn), shape (NUM_ENVS, 3)."""
    generator = torch.Generator().manual_seed(1)
    rest = torch.zeros(NUM_ENVS, 3, dtype=torch.float64)
    rest[:, 0] = 0.50 + 0.15 * torch.rand(NUM_ENVS, generator=generator, dtype=torch.float64)
    rest[:, 1] = -0.15 + 0.30 * torch.rand(NUM_ENVS, generator=generator, dtype=torch.float64)
    rest[:, 2] = ARC_LENGTH
    return rest


def _sample(
    angle_range: tuple[float, float] = FULL_CIRCLE,
    seed: int = 0,
    arc_length: float = ARC_LENGTH,
    distance_range: tuple[float, float] = DISTANCE_RANGE,
    max_curvature: float = CURVATURE_AT_BUDGET,
) -> tuple[torch.Tensor, torch.Tensor]:
    rest = _rest_points()
    targets = stem_target.sample_targets(
        rest,
        arc_length,
        distance_range,
        angle_range,
        max_curvature,
        generator=torch.Generator().manual_seed(seed),
    )
    return rest, targets


def _distance_and_drop(rest: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal distance r [m] and drop [m] of each target below its rest position."""
    return (targets[:, :2] - rest[:, :2]).norm(dim=-1), rest[:, 2] - targets[:, 2]


@pytest.mark.parametrize("distance", [0.03, 0.05, 0.075, 0.10])
def test_bowl_matches_numerical_integration(distance: float):
    table = stem_target.rod_table(BUDGET)
    drop = float(table.bowl_drop(torch.tensor([distance], dtype=torch.float64), ARC_LENGTH)[0])
    assert drop == pytest.approx(_bowl(distance), abs=1e-5)  # [m], 0.01 mm


@pytest.mark.parametrize("distance", list(DEEPEST_SHAPES))
def test_deepest_shape_matches_numerical_integration(distance: float):
    """The optimizer's shape reaches r within the budget; the table's deepest drop is that shape's drop."""
    theta, kappa, x, z = _shoot(*DEEPEST_SHAPES[distance])
    assert abs(x[-1]) * ARC_LENGTH == pytest.approx(distance, abs=1e-6)
    assert float(np.abs(kappa).max()) <= BUDGET * (1 + 1e-5)
    table = stem_target.rod_table(BUDGET)
    drop = float(table.deepest_drop(torch.tensor([distance], dtype=torch.float64), ARC_LENGTH)[0])
    assert drop == pytest.approx((1.0 - z[-1]) * ARC_LENGTH, abs=2e-5)  # [m], 0.02 mm


def test_room_below_the_bowl():
    """Room between the bowl and the deepest shape at the decided budget: 14.1 mm at r = 3 cm, 18.0 mm at 10 cm
    (independent integration above; the linear small-deflection model gave 5.5 and 8.3 mm)."""
    table = stem_target.rod_table(BUDGET)
    distance = torch.tensor([0.03, 0.10], dtype=torch.float64)
    room = table.deepest_drop(distance, ARC_LENGTH) - table.bowl_drop(distance, ARC_LENGTH)
    torch.testing.assert_close(room, torch.tensor([0.01408, 0.01798], dtype=torch.float64), rtol=0.0, atol=5e-5)


def test_distance_beyond_the_table_is_rejected():
    table = stem_target.rod_table(BUDGET)
    with pytest.raises(ValueError):
        table.bowl_drop(torch.tensor([0.30], dtype=torch.float64), ARC_LENGTH)


def test_targets_lie_in_the_region():
    rest, targets = _sample()
    assert targets.shape == (NUM_ENVS, 3)
    distance, drop = _distance_and_drop(rest, targets)
    assert float(distance.min()) >= DISTANCE_RANGE[0] - 1e-9
    assert float(distance.max()) <= DISTANCE_RANGE[1] + 1e-9
    table = stem_target.rod_table(BUDGET)
    bowl, deepest = table.bowl_drop(distance, ARC_LENGTH), table.deepest_drop(distance, ARC_LENGTH)
    assert float((drop - bowl).min()) >= -1e-9  # on or below the bowl
    assert float((drop - deepest).max()) <= 1e-9  # within the budget
    # the whole band is used, uniformly: targets near the bowl and near the deepest shape
    position = (drop - bowl) / (deepest - bowl)
    counts = torch.histc(position, bins=8, min=0.0, max=1.0)
    assert bool((counts > 0.8 * NUM_ENVS / 8).all()), counts


def test_distances_are_uniform():
    rest, targets = _sample()
    distance, _ = _distance_and_drop(rest, targets)
    counts = torch.histc(distance, bins=7, min=DISTANCE_RANGE[0], max=DISTANCE_RANGE[1])
    assert bool((counts > 0.8 * NUM_ENVS / 7).all()), counts


def test_directions_cover_the_angle_range():
    rest, targets = _sample()
    angle = torch.atan2(targets[:, 1] - rest[:, 1], targets[:, 0] - rest[:, 0])
    counts = torch.histc(angle, bins=8, min=-math.pi, max=math.pi)
    assert bool((counts > 0.8 * NUM_ENVS / 8).all()), counts  # about uniform over the full circle

    quarter = (0.0, 0.5 * math.pi)
    rest, targets = _sample(angle_range=quarter)
    angle = torch.atan2(targets[:, 1] - rest[:, 1], targets[:, 0] - rest[:, 0])
    assert float(angle.min()) >= quarter[0] - 1e-6 and float(angle.max()) <= quarter[1] + 1e-6


def test_targets_scale_with_the_arc_length():
    """A stem twice as long, with the same budget x s and distances scaled with it: offsets twice as large."""
    rest, short = _sample()
    _, long = _sample(arc_length=2 * ARC_LENGTH, distance_range=(0.06, 0.20), max_curvature=0.5 * CURVATURE_AT_BUDGET)
    torch.testing.assert_close(long - rest, 2 * (short - rest), rtol=1e-9, atol=1e-12)


def test_targets_differ_per_env_and_per_call():
    rest, first = _sample(seed=0)
    _, second = _sample(seed=1)
    offsets = first - rest
    assert torch.unique(offsets, dim=0).shape[0] == NUM_ENVS
    assert not torch.allclose(first, second)
