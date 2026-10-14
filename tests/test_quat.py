"""Unit tests of the torch-only quaternion module (x, y, z, w); no simulator needed."""

import math
import subprocess
import sys

import pytest
import torch

from stem_manip.utils.quat import quat_apply, quat_conjugate, quat_from_rotvec, quat_mul, rotvec_from_quat

N = 64
DTYPE = torch.float64


def _random_quats(generator: torch.Generator) -> torch.Tensor:
    q = torch.randn(N, 4, dtype=DTYPE, generator=generator)
    return q / q.norm(dim=-1, keepdim=True)


def test_mul_identity_and_conjugate():
    q = _random_quats(torch.Generator().manual_seed(0))
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DTYPE).expand(N, 4)
    assert torch.allclose(quat_mul(q, identity), q)
    assert torch.allclose(quat_mul(q, quat_conjugate(q)), identity)


def test_apply_quarter_turn_about_z():
    q = quat_from_rotvec(torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=DTYPE))
    rotated = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE))
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE), atol=1e-12)


def test_rotvec_round_trip_and_zero():
    generator = torch.Generator().manual_seed(1)
    rotvec = torch.randn(N, 3, dtype=DTYPE, generator=generator)
    rotvec = rotvec / rotvec.norm(dim=-1, keepdim=True) * torch.rand(N, 1, dtype=DTYPE, generator=generator) * 3.0
    assert torch.allclose(rotvec_from_quat(quat_from_rotvec(rotvec)), rotvec, atol=1e-9)
    zero = torch.zeros(1, 3, dtype=DTYPE)
    assert torch.equal(quat_from_rotvec(zero), torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=DTYPE))
    assert torch.equal(rotvec_from_quat(quat_from_rotvec(zero)), zero)


def test_rotvec_from_quat_takes_the_short_way():
    q = quat_from_rotvec(torch.tensor([[0.0, 0.0, 0.4]], dtype=DTYPE))
    assert torch.allclose(rotvec_from_quat(-q), rotvec_from_quat(q))


def test_matches_isaaclab_math():
    math_lab = pytest.importorskip("isaaclab.utils.math")
    generator = torch.Generator().manual_seed(2)
    a, b = _random_quats(generator), _random_quats(generator)
    v = torch.randn(N, 3, dtype=DTYPE, generator=generator)
    assert torch.allclose(quat_mul(a, b), math_lab.quat_mul(a, b))
    assert torch.allclose(quat_conjugate(a), math_lab.quat_conjugate(a))
    assert torch.allclose(quat_apply(a, v), math_lab.quat_apply(a, v))


def test_rotvec_matches_isaaclab_math():
    math_lab = pytest.importorskip("isaaclab.utils.math")
    generator = torch.Generator().manual_seed(3)
    axis = torch.randn(N, 3, dtype=DTYPE, generator=generator)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = 0.05 + 2.9 * torch.rand(N, dtype=DTYPE, generator=generator)  # away from pi, where the sign is ambiguous
    rotvec = axis * angle.unsqueeze(-1)
    q = quat_from_rotvec(rotvec)
    assert torch.allclose(q, math_lab.quat_from_angle_axis(angle, axis))
    assert torch.allclose(rotvec_from_quat(q), math_lab.axis_angle_from_quat(q))


def test_stem_geometry_import_loads_no_isaaclab():
    code = "import sys, stem_manip.utils.stem_geometry; print([m for m in sys.modules if m.startswith('isaaclab')])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
