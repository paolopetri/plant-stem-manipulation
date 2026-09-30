"""Isaac Lab config for the plant stem, modelled as a Newton cable (stem model v1).

The stem is a vertical chain of capsule segments joined by cable joints (see IsaacLab
`docs/source/concepts/deformables.rst`, section Cables). All parameters come from `assets/stem/stem.yaml`.
Newton only (VBD solver): the scene must use a Newton physics cfg.

Not done yet (docs/TODO.md -> M1):
- Damping (`material.damping_time` is not applied).
- The stem base is not held fixed: the root segment is free-floating. Pinning only gives a ball joint, so try
  pinning the first two control points. Document the chosen workaround here.

Verify: `scripts/check_stem.py`.
"""

import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObjectCfg

from stem_manip.assets import REPO_ASSETS_DIR

STEM_YAML = REPO_ASSETS_DIR / "stem" / "stem.yaml"


def stem_params() -> dict:
    """Nominal stem parameters (geometry, material, damage limits) from `assets/stem/stem.yaml`."""
    return yaml.safe_load(STEM_YAML.read_text())


def stem_cfg() -> CableObjectCfg:
    """Vertical stem with its base at `geometry.base_position` in the env frame. Set `prim_path` in the scene."""
    params = stem_params()
    geometry, material = params["geometry"], params["material"]
    segment_length = geometry["length"] / geometry["num_segments"]
    return CableObjectCfg(
        spawn=sim_utils.CableCfg(
            positions=[(0.0, 0.0, index * segment_length) for index in range(geometry["num_segments"] + 1)],
            physics_material=sim_utils.CableMaterialCfg(
                thickness=geometry["diameter"],
                density=material["density"],
                stretch_stiffness=material["stretch_modulus"],
                bend_stiffness=material["bend_modulus"],
                twist_stiffness=material["twist_modulus"],
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=tuple(geometry["base_position"])),
    )
