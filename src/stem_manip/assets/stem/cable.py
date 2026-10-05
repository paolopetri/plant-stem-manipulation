"""Stem model `cable`: the plant stem as a Newton cable.

The stem is a vertical chain of capsule segments joined by cable joints (see IsaacLab
`docs/source/concepts/deformables.rst`, section Cables). Parameters: `stem_params("cable")`, i.e.
`assets/stem/stem.yaml` + `assets/stem/cable/cable.yaml`. Newton only (VBD solver): the scene must use
`physics_cfg()`.

Damping: Isaac Lab's `CableMaterialCfg` has no damping fields, but Newton reads four per-mode values from the
material prim (`newton:curves{Stretch,Shear,Bend,Twist}Damping`, schema `NewtonCurvesDeformableMaterialAPI`).
`StemMaterialCfg` adds them. They are structural values, set stiffness-proportional from one time constant:
damping = `material.damping_time` * structural stiffness (E I for bend, G J for twist). Stretch and shear are not
damped: their damping slows the solver's convergence and makes affordable solver settings unstable.

Fixed base: the cable's root segment is free-floating, and Isaac Lab only offers pins (ball joints), which do
not clamp. `fix_stem_base()` instead marks the root segment as a kinematic body in the Newton model: the solver
skips it, so it keeps its pose, and the second segment is tied to it by the normal cable joint (bend, twist,
stretch). This is a full clamp; the lowest segment no longer deforms, so the flexible length is
`length * (1 - 1 / num_segments)`. Call it after the simulation is built (the flag lives in the Newton model,
not in the cfg).

Verify: `scripts/check_stem.py`.
"""

import math
from typing import ClassVar

import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonManager, VBDSolverCfg
from newton import BodyFlags, ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObject, CableObjectCfg
from isaaclab.utils import configclass

from stem_manip.assets.stem import stem_params


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


def physics_cfg() -> NewtonCfg:
    """Newton VBD physics cfg with the settings of `solver` in `cable.yaml` (the step `solver.sim_dt` goes into
    the `SimulationCfg`)."""
    solver = stem_params("cable")["solver"]
    return NewtonCfg(solver_cfg=VBDSolverCfg(iterations=solver["vbd_iterations"]), num_substeps=solver["num_substeps"])


def stem_cfg() -> CableObjectCfg:
    """Vertical stem with its base at `geometry.base_position` in the env frame. Set `prim_path` in the scene."""
    params = stem_params("cable")
    geometry, material = params["geometry"], params["material"]
    segment_length = geometry["length"] / geometry["num_segments"]

    # structural bend and twist stiffness as Newton derives them from the moduli (twist = bend if not set)
    area_moment = math.pi * geometry["diameter"] ** 4 / 64
    bend_stiffness = material["bend_modulus"] * area_moment
    twist_modulus = material["twist_modulus"]
    twist_stiffness = bend_stiffness if twist_modulus is None else twist_modulus * 2.0 * area_moment
    damping_time = material["damping_time"]
    damping = {}
    if damping_time is not None:
        damping = {
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
                shear_stiffness=material["shear_modulus"],
                bend_stiffness=material["bend_modulus"],
                twist_stiffness=twist_modulus,
                **damping,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=tuple(geometry["base_position"])),
    )


def fix_stem_base(stem: CableObject) -> None:
    """Clamp the stem base: make the root segment of every env a kinematic body (see the module docstring).

    Call after `sim.reset()` / scene creation, and again if the Newton model is rebuilt.
    """
    model = NewtonManager.get_model()
    root_body_ids = stem.root_view.get_attribute("joint_parent", model).numpy()[:, 0, 0]
    body_flags = model.body_flags.numpy()
    body_flags[root_body_ids] = int(BodyFlags.KINEMATIC)
    model.body_flags.assign(body_flags)
    NewtonManager.add_model_change(ModelFlags.BODY_PROPERTIES)


@wp.kernel
def _add_body_forces(forces: wp.array(dtype=wp.spatial_vectorf), body_f: wp.array(dtype=wp.spatial_vectorf)):
    body = wp.tid()
    body_f[body] = body_f[body] + forces[body]


def register_body_forces() -> wp.array:
    """External forces on the bodies of the Newton model, e.g. to push a stem segment in a check script.

    Returns an array with one entry per body of the model: force (x, y, z) [N] and torque (x, y, z) [N m] in the
    world frame, acting at the body's centre of mass. It starts at zero; write into it (`assign`, `zero_`) to
    switch forces on and off. Newton clears the body forces after every solver substep, so a callback re-applies
    the array before each substep. The callback is recorded with the simulation step (CUDA graph): call this
    after the simulation is built and before the first step.
    """
    model = NewtonManager.get_model()
    forces = wp.zeros(model.body_count, dtype=wp.spatial_vectorf, device=model.device)
    NewtonManager.register_state_force_callback(
        lambda state: wp.launch(
            _add_body_forces, dim=model.body_count, inputs=[forces], outputs=[state.body_f], device=model.device
        )
    )
    return forces
