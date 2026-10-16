"""Target positions for the stem point (stage 1): on or below the "bowl" it reaches when pushed sideways, down to the
deepest shape within a curvature budget (user, 2026-10-08/09; spec #160, ADR 0004).

The stem does not stretch, so a point at arc length s pushed sideways by r also drops. Shape model: a planar
elastica, an inextensible rod clamped upright at its base with a force and a moment at the point (large
deflection). With EI = 1 its angle from the vertical theta obeys theta'' = fx cos(theta) - fz sin(theta) (fz > 0
presses down). The shape depends on the stem's length and curvature limit, not on its stiffness (stiffness sets the
force needed), so the reachable region is dimensionless: r / s, drop / s, curvature budget x s. The rod table
(`rod_table`) holds, per r / s:
- the bowl: the drop of the stem pushed sideways at the point by a horizontal force alone (no moment). Closed form
  of that elastica, integrated by quadrature;
- the deepest shape: the largest drop of any elastica whose curvature stays within the budget. A moment or a C / S
  bend reaches it, and in large deflection also pressing the point down (axial compression).
The deepest shape has its largest curvature, equal to the budget, where the internal force is along the stem
(kappa' = 0; checked against a constrained optimizer over all (k0, fx, fz), `tests/test_stem_target.py`). In the
frame of that point the shape depends only on the force P: kappa^2 = budget^2 - 2 P (1 - cos(phi)) <= budget^2.
So the table integrates one curve per P from that point both ways and takes every window of length 1 along it;
the deepest drop at each r / s is the largest over all P and windows.
Nothing above the bowl is sampled (user, 2026-10-09).

Assumes an upright stem in a frame with z up (robot base frame). Batched over envs, no simulator needed.
Tests: `tests/test_stem_target.py`.
"""

import functools
from dataclasses import dataclass

import torch

DISTANCE_NODES = 101
"""Table nodes in r / s, uniform from 0 to `DISTANCE_LIMIT`."""
DISTANCE_LIMIT = 0.5
"""Largest r / s in the table; the table ends earlier where the bowl itself exceeds the budget."""
FORCE_MAX, FORCE_SAMPLES = 100.0, 1001
"""Force P (EI = 1, s = 1) of the deepest shapes, sampled uniformly from 0."""
STEPS = 400
"""Integration steps per stem length (RK4)."""


@dataclass(frozen=True)
class RodTable:
    """Reachable region of the stem point, dimensionless (r / s, drop / s), for one curvature budget x s.
    CPU, float64; the drop functions take and return tensors on any device."""

    curvature_budget: float
    """Curvature budget x arc length s [-]."""
    distance: torch.Tensor
    """Nodes r / s, uniform from 0."""
    bowl: torch.Tensor
    """Drop / s of the sideways push at each node."""
    deepest: torch.Tensor
    """Drop / s of the deepest shape within the budget at each node."""
    max_distance: float
    """Largest r / s the table covers: the bowl's own curvature stays within the budget up to here."""

    def bowl_drop(self, distance: torch.Tensor, arc_length: float) -> torch.Tensor:
        """Drop [m] of the point at `arc_length` [m] pushed sideways by `distance` [m] (the bowl)."""
        return self._lookup(self.bowl, distance, arc_length)

    def deepest_drop(self, distance: torch.Tensor, arc_length: float) -> torch.Tensor:
        """Largest drop [m] within the budget of the point at `arc_length` [m] at sideways `distance` [m]."""
        return self._lookup(self.deepest, distance, arc_length)

    def _lookup(self, values: torch.Tensor, distance: torch.Tensor, arc_length: float) -> torch.Tensor:
        x = distance / arc_length
        if float(x.max()) > self.max_distance:
            raise ValueError(f"r / s up to {float(x.max()):.3f}, the rod table covers {self.max_distance:.3f}")
        nodes, values = self.distance.to(x), values.to(x)
        return _interp(x, nodes, values) * arc_length


@functools.lru_cache
def rod_table(curvature_budget: float) -> RodTable:
    """Rod table for `curvature_budget` = max curvature x s [-], computed once per budget (cached)."""
    nodes = torch.linspace(0.0, DISTANCE_LIMIT, DISTANCE_NODES, dtype=torch.float64)
    bowl, bowl_curvature, bowl_distance = _bowl(nodes)
    # the bowl's base curvature grows with r; past the budget no target lies in the region
    over = bowl_curvature > curvature_budget
    max_distance = float(bowl_distance[over][0]) if bool(over.any()) else DISTANCE_LIMIT
    return RodTable(
        curvature_budget=curvature_budget,
        distance=nodes,
        bowl=_interp(nodes, bowl_distance, bowl),
        deepest=_deepest(nodes, curvature_budget),
        max_distance=min(max_distance, DISTANCE_LIMIT),
    )


def _interp(x: torch.Tensor, nodes: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of `values` at increasing `nodes`, at `x`."""
    i = torch.searchsorted(nodes, x.contiguous()).clamp(1, nodes.numel() - 1)
    w = (x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])
    return values[i - 1] + w * (values[i] - values[i - 1])


def _bowl(nodes: torch.Tensor, samples: int = 2000, quadrature: int = 4001) -> tuple[torch.Tensor, ...]:
    """Sideways push (horizontal force P, no moment), s = 1: kappa^2 = 2 P (sin(theta_1) - sin(theta)) with tip angle
    theta_1. With sin(theta) = a (1 - u^2), a = sin(theta_1), u in [0, 1]: s = sqrt(2 a / P) int sec(theta) du = 1,
    x = int tan(theta) du / I0, z = 1 / I0 with I0 = int sec(theta) du, base curvature 2 a I0.
    Returns (drop, base curvature, r) per sampled tip angle, r increasing and beyond the last node."""
    a = torch.sin(torch.linspace(1e-4, 1.4, samples, dtype=torch.float64))[:, None]
    u = torch.linspace(0.0, 1.0, quadrature, dtype=torch.float64)
    sin = a * (1.0 - u**2)
    cos = (1.0 - sin**2).sqrt()
    i0 = torch.trapezoid(1.0 / cos, u, dim=-1)
    distance = torch.trapezoid(sin / cos, u, dim=-1) / i0
    assert float(distance[-1]) > float(nodes[-1])
    return 1.0 - 1.0 / i0, 2.0 * a[:, 0] * i0, distance


def _deepest(nodes: torch.Tensor, curvature_budget: float) -> torch.Tensor:
    """Deepest drop / s within `curvature_budget` at each node r / s, over all elastica with the budget as their
    largest curvature (module docstring)."""
    force = torch.linspace(0.0, FORCE_MAX, FORCE_SAMPLES, dtype=torch.float64)
    h = 1.0 / STEPS

    def rate(state: torch.Tensor) -> torch.Tensor:  # (phi, kappa, x, z), frame of the largest curvature
        phi, kappa = state[0], state[1]
        return torch.stack((kappa, -force * torch.sin(phi), torch.sin(phi), torch.cos(phi)))

    def integrate(step: float) -> torch.Tensor:
        state = torch.zeros(4, force.numel(), dtype=torch.float64)
        state[1] = curvature_budget
        states = [state]
        for _ in range(STEPS):
            k1 = rate(state)
            k2 = rate(state + 0.5 * step * k1)
            k3 = rate(state + 0.5 * step * k2)
            k4 = rate(state + step * k3)
            state = state + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            states.append(state)
        return torch.stack(states, dim=-1)  # (4, forces, STEPS + 1)

    curve = torch.cat((integrate(-h).flip(-1)[..., :-1], integrate(h)), dim=-1)  # arc length -1 .. 1
    base, tip = curve[..., : STEPS + 1], curve[..., STEPS:]  # windows of length 1 holding the largest curvature
    phi = base[0]
    dx, dz = tip[2] - base[2], tip[3] - base[3]
    x = dx * torch.cos(phi) - dz * torch.sin(phi)  # rotated so that the base is upright
    drop = 1.0 - (dx * torch.sin(phi) + dz * torch.cos(phi))
    x, drop = torch.cat((x, -x)), torch.cat((drop, drop))  # mirrored shapes reach the same r

    # each step between neighbouring windows crosses at most one node (it moves r / s by at most 1 / STEPS, less
    # than the node spacing): interpolate the drop there and keep the largest per node
    spacing = float(nodes[1] - nodes[0])
    x0, x1, d0, d1 = x[:, :-1].flatten(), x[:, 1:].flatten(), drop[:, :-1].flatten(), drop[:, 1:].flatten()
    node = torch.ceil(torch.minimum(x0, x1) / spacing).long()
    crosses = (node * spacing <= torch.maximum(x0, x1)) & (node >= 0) & (node < nodes.numel()) & (x0 != x1)
    node, x0, x1, d0, d1 = node[crosses], x0[crosses], x1[crosses], d0[crosses], d1[crosses]
    value = d0 + (nodes[node] - x0) / (x1 - x0) * (d1 - d0)
    deepest = torch.full_like(nodes, -torch.inf)
    return deepest.scatter_reduce(0, node, value, reduce="amax")


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
        max_curvature: Curvature budget of the deepest target shape [1/m]; the drop is sampled uniformly between
            the bowl and the deepest shape (`rod_table`).
        generator: Random generator on the device of `rest_points` (reproducible tests); `None` = torch's default.

    Returns:
        The targets, in the frame of `rest_points`.
    """
    num_envs, device = rest_points.shape[0], rest_points.device
    table = rod_table(max_curvature * arc_length)

    def uniform(low: float | torch.Tensor, high: float | torch.Tensor) -> torch.Tensor:
        u = torch.rand(num_envs, generator=generator, device=device, dtype=rest_points.dtype)
        return low + (high - low) * u

    distance = uniform(*distance_range)
    angle = uniform(*angle_range)
    drop = uniform(table.bowl_drop(distance, arc_length), table.deepest_drop(distance, arc_length))
    offset = torch.stack((distance * torch.cos(angle), distance * torch.sin(angle), -drop), dim=-1)
    return rest_points + offset
