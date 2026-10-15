"""Quaternion math (x, y, z, w), torch only, as `isaaclab.utils.math` but without the simulator.

Shared by `impedance.py` (which is ported 1:1 to the real FR3) and `stem_geometry.py`. Batched over leading dims.
Tests: `tests/test_quat.py`.
"""

from __future__ import annotations

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b, quaternions (x, y, z, w)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (inverse of a unit quaternion), quaternions (x, y, z, w)."""
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (..., 4) (x, y, z, w) of rotation vectors (..., 3) (axis * angle [rad])."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(a/2)/a -> 1/2 for a -> 0 (series), keeps small and zero rotations exact
    scale = torch.where(angle > 1e-6, torch.sin(half) / angle.clamp(min=1e-12), 0.5 - angle**2 / 48.0)
    return torch.cat((rotvec * scale, torch.cos(half)), dim=-1)


def rotvec_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Rotation vectors (..., 3) (axis * angle [rad], angle in [0, pi]) of unit quaternions (..., 4) (x, y, z, w)."""
    quat = torch.where(quat[..., 3:] < 0.0, -quat, quat)  # w >= 0: the shorter rotation
    sin_half = quat[..., :3].norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, quat[..., 3:])
    # angle / sin(a/2) -> 2 for a -> 0, keeps small and zero rotations exact
    scale = torch.where(sin_half > 1e-6, angle / sin_half.clamp(min=1e-12), 2.0 + angle**2 / 12.0)
    return quat[..., :3] * scale


def rotation_angle(quat_from: torch.Tensor, quat_to: torch.Tensor) -> torch.Tensor:
    """Angle (...,) [rad] in [0, pi] of the rotation from `quat_from` to `quat_to`, unit quaternions (..., 4)
    (x, y, z, w) in the same frame."""
    return rotvec_from_quat(quat_mul(quat_to, quat_conjugate(quat_from))).norm(dim=-1)
