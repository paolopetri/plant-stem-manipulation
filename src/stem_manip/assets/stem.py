"""Isaac Lab config for the plant stem, modelled as a Newton cable (stem model v1).

The stem is a vertical chain of capsule segments joined by cable joints (see IsaacLab
`docs/source/concepts/deformables.rst`, section Cables). All parameters come from `assets/stem/stem.yaml`.
Newton only (VBD solver): the scene must use a Newton physics cfg.

Damping: Isaac Lab's `CableMaterialCfg` has no damping fields, but Newton reads four per-mode values from the
material prim (`newton:curves{Stretch,Shear,Bend,Twist}Damping`, schema `NewtonCurvesDeformableMaterialAPI`).
`StemMaterialCfg` adds them. They are structural values, set stiffness-proportional from one time constant:
damping = `material.damping_time` * structural stiffness (E A for stretch and shear, E I for bend, G J for twist).

Not done yet (docs/TODO.md -> M1):
- The stem base is not held fixed: the root segment is free-floating. Pinning only gives a ball joint, so try
  pinning the first two control points. Document the chosen workaround here.

Verify: `scripts/check_stem.py`.
"""

import math
from typing import ClassVar

import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObjectCfg
from isaaclab.utils import configclass

from stem_manip.assets import REPO_ASSETS_DIR

STEM_YAML = REPO_ASSETS_DIR / "stem" / "stem.yaml"


@configclass
class StemMaterialCfg(sim_utils.CableMaterialCfg):
    """Cable material with Newton's per-mode damping, which Isaac Lab's `CableMaterialCfg` does not expose.

    Isaac Lab writes each field under the namespace of the class that declares it, so the fields below become
    `newton:curves*Damping` on the material prim; the inherited stiffness fields stay under `physics:`.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonCurvesDeformableMaterialAPI"

    curves_stretch_damping: float | None = None
    """Stretch damping [N s]. None = not authored (Newton default 0)."""

    curves_shear_damping: float | None = None
    """Transverse shear damping [N s]. None = not authored (Newton default 0)."""

    curves_bend_damping: float | None = None
    """Bend damping [N m^2 s]. None = not authored (Newton default 0)."""

    curves_twist_damping: float | None = None
    """Twist damping [N m^2 s]. None = not authored (Newton default 0)."""


def stem_params() -> dict:
    """Nominal stem parameters (geometry, material, damage limits) from `assets/stem/stem.yaml`."""
    return yaml.safe_load(STEM_YAML.read_text())


def stem_cfg() -> CableObjectCfg:
    """Vertical stem with its base at `geometry.base_position` in the env frame. Set `prim_path` in the scene."""
    params = stem_params()
    geometry, material = params["geometry"], params["material"]
    segment_length = geometry["length"] / geometry["num_segments"]

    # structural stiffnesses as Newton derives them from the moduli (shear = stretch, twist = bend if not set)
    area = math.pi * geometry["diameter"] ** 2 / 4
    area_moment = math.pi * geometry["diameter"] ** 4 / 64
    stretch_stiffness = material["stretch_modulus"] * area
    bend_stiffness = material["bend_modulus"] * area_moment
    twist_modulus = material["twist_modulus"]
    twist_stiffness = bend_stiffness if twist_modulus is None else twist_modulus * 2.0 * area_moment
    damping_time = material["damping_time"]
    damping = {}
    if damping_time is not None:
        damping = {
            "curves_stretch_damping": damping_time * stretch_stiffness,
            "curves_shear_damping": damping_time * stretch_stiffness,
            "curves_bend_damping": damping_time * bend_stiffness,
            "curves_twist_damping": damping_time * twist_stiffness,
        }

    return CableObjectCfg(
        spawn=sim_utils.CableCfg(
            positions=[(0.0, 0.0, index * segment_length) for index in range(geometry["num_segments"] + 1)],
            physics_material=StemMaterialCfg(
                thickness=geometry["diameter"],
                density=material["density"],
                stretch_stiffness=material["stretch_modulus"],
                bend_stiffness=material["bend_modulus"],
                twist_stiffness=twist_modulus,
                **damping,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=tuple(geometry["base_position"])),
    )
