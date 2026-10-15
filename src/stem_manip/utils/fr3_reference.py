"""Reference slot-base kinematics for the checks: FR3 forward kinematics from the datasheet and the yaml slot base.

Independent of the slot-base code of the task (`mdp/observations.py`, `mdp/actions.py`, the URDF/USD and
`stem_manip.assets.fr3.slot_base_offset`) on purpose: the checks compare the task's slot-base pose with it, so the
task's code must not use it. Joint angles -> fr3_link8 (flange) pose via Franka's modified Denavit-Hartenberg
parameters (Craig), then the end-effector yaml's `slot_base` (xyz, URDF rpy), which the yaml gives in the fr3_link8
frame. Torch only, float64; poses as 4x4 matrices in the fr3_link0 (robot base) frame.
Used by `scripts/check_fr3.py` and `scripts/check_push_env.py`. Tests: `tests/test_fr3_reference.py`.
"""

import math
from collections.abc import Sequence

import torch
import yaml

from stem_manip.assets import REPO_ASSETS_DIR

DTYPE = torch.float64
# FR3 modified DH parameters (a [m], d [m], alpha [rad]) of joints 1-7 and the flange (Franka's robot and interface
# specifications, "Denavit-Hartenberg parameters"); per row: T = RotX(alpha) TransX(a) RotZ(theta) TransZ(d)
DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -math.pi / 2),
    (0.0, 0.316, math.pi / 2),
    (0.0825, 0.0, math.pi / 2),
    (-0.0825, 0.384, -math.pi / 2),
    (0.0, 0.0, math.pi / 2),
    (0.088, 0.0, math.pi / 2),
)
FLANGE = (0.0, 0.107, 0.0)  # fr3_link7 -> fr3_link8, theta = 0


def _dh_transform(a: float, d: float, alpha: float, theta: torch.Tensor) -> torch.Tensor:
    """RotX(alpha) TransX(a) RotZ(theta) TransZ(d), batched over theta (n,) -> (n, 4, 4)."""
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    rows = (
        (ct, -st, zero, a * one),
        (st * ca, ct * ca, -sa * one, -sa * d * one),
        (st * sa, ct * sa, ca * one, ca * d * one),
        (zero, zero, zero, one),
    )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def flange_pose(q: torch.Tensor) -> torch.Tensor:
    """fr3_link8 (flange) pose (n, 4, 4) in the fr3_link0 frame for the joint angles q (n, 7) [rad]."""
    q = q.to(DTYPE)
    pose = torch.eye(4, dtype=DTYPE, device=q.device).repeat(q.shape[0], 1, 1)
    for (a, d, alpha), theta in zip(DH, q.unbind(-1), strict=True):
        pose = pose @ _dh_transform(a, d, alpha, theta)
    return pose @ _dh_transform(*FLANGE, torch.zeros_like(q[:, 0]))


def rpy_matrix(rpy: Sequence[float]) -> torch.Tensor:
    """Rotation matrix (3, 3) of URDF rpy: fixed-axis roll, pitch, yaw, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=DTYPE)
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=DTYPE)
    rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE)
    return rz @ ry @ rx


def slot_base_in_flange(ee: str) -> torch.Tensor:
    """Slot-base pose (4, 4) in the fr3_link8 frame, from `slot_base` of assets/fr3/end_effectors/<ee>/<ee>.yaml.

    Raises ValueError unless the yaml's `mount` is the identity: the yaml gives `slot_base` in the fr3_link8 frame,
    the URDF builder (and so the task) puts it under the end-effector body; the two agree only with an identity mount.
    """
    path = REPO_ASSETS_DIR / "fr3" / "end_effectors" / ee / f"{ee}.yaml"
    description = yaml.safe_load(path.read_text())
    mount = description["mount"]
    if any(v != 0 for v in (*mount["xyz"], *mount["rpy"])):
        raise ValueError(f"{path.name}: mount {mount} is not the identity; slot_base in fr3_link8 no longer holds")
    slot_base = description["slot_base"]
    pose = torch.eye(4, dtype=DTYPE)
    pose[:3, :3] = rpy_matrix(slot_base["rpy"])
    pose[:3, 3] = torch.tensor(slot_base["xyz"], dtype=DTYPE)
    return pose


def slot_base_pose(q: torch.Tensor, ee: str) -> torch.Tensor:
    """Slot-base pose (n, 4, 4) in the fr3_link0 frame for the joint angles q (n, 7) [rad] and end-effector `ee`."""
    return flange_pose(q) @ slot_base_in_flange(ee).to(q.device)


def rotvec_from_matrix(rot: torch.Tensor) -> torch.Tensor:
    """Rotation vector (n, 3) [rad] of rotation matrices (n, 3, 3), for angles below pi.

    The skew part of R is sin(angle) * axis; the angle comes from atan2, so small angles stay exact.
    """
    v = 0.5 * torch.stack(
        (rot[:, 2, 1] - rot[:, 1, 2], rot[:, 0, 2] - rot[:, 2, 0], rot[:, 1, 0] - rot[:, 0, 1]), dim=-1
    )
    sin = v.norm(dim=-1)
    cos = 0.5 * (rot.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    angle = torch.atan2(sin, cos)
    scale = torch.where(sin > 1e-12, angle / sin.clamp(min=1e-12), torch.ones_like(sin))
    return v * scale.unsqueeze(-1)
