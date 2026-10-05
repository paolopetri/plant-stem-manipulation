"""Stem model `chain`: the plant stem as a PhysX articulation of rigid capsule segments with spring joints.

Segment 0 is clamped to the world by a fixed joint. Every other joint connects two neighbouring segments at their
common end and only rotates (D6 joint, translations locked; PhysX treats it as a spherical joint with 3 DOFs,
named `joint_<i>:0` twist, `:1` and `:2` bend). Each rotation has a spring and a damper, set as Isaac Lab
implicit actuators: bend E I / l, twist G J / l, damping = `material.damping_time` * stiffness, plus the joint
`armature` of `chain.yaml`. The chain cannot stretch or shear by construction. Parameters: `stem_params("chain")`,
i.e. `assets/stem/stem.yaml` + `assets/stem/chain/chain.yaml`. PhysX only: the scene must use `physics_cfg()`.

Collision surface: a smooth tube of the stem's radius around its centre line. Each segment's capsule has the
segment length as its cylinder part, so its round ends are centred on the joints; at every joint the neighbours
share the same sphere, and the union stays smooth at any bend angle (capsules that only touch would leave a groove
of depth radius at every joint, where an edge of the tool can catch). Neighbouring segments do not collide with
each other. The end segments are shortened by half a radius so that the tube ends at the base and at the tip.

The USD holds only the geometry (segments, collision capsules, masses, joint frames) and is written from the yaml
at spawn time into `assets/stem/chain/build/` (file name = hash of the values it depends on). Stiffness, damping,
armature and masses are runtime properties: domain randomization writes them per env
(`write_joint_stiffness_to_sim_index`, `set_masses_index`, ...) without rebuilding the USD. Length and diameter
change the geometry and need one USD per variant, chosen per env when the scene is created.

Verify: `scripts/check_stem.py --stem_model chain`.
"""

import hashlib
import json
import math
import os
from collections.abc import Callable

import numpy as np
import torch

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils import configclass

from stem_manip.assets.stem import STEM_DIR, stem_params

BUILD_DIR = STEM_DIR / "chain" / "build"


def write_chain_usd(path: str, num_segments: int, length: float, diameter: float, density: float) -> None:
    """Write the chain USD: an upright stem with its base at the origin, segment i centred at (i + 1/2) l."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # needs the running app (Kit's USD), so imported here

    segment_length, radius = length / num_segments, diameter / 2
    mass = density * math.pi * radius**2 * segment_length
    inertia_bend = mass * (3 * radius**2 + segment_length**2) / 12  # solid cylinder about its centre
    # joint frame: x axis (twist) along the stem axis (segment z), y and z bend
    joint_rot = Gf.Quatf(math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Stem")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    for i in range(num_segments):
        segment = UsdGeom.Xform.Define(stage, f"/Stem/seg_{i:02d}")
        segment.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, (i + 0.5) * segment_length))
        UsdPhysics.RigidBodyAPI.Apply(segment.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(segment.GetPrim())
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia_bend, inertia_bend, mass * radius**2 / 2))
        # round ends centred on the joints (see the module docstring); the end segments stop at the base / the tip
        capsule = UsdGeom.Capsule.Define(stage, f"/Stem/seg_{i:02d}/collision")
        capsule.CreateAxisAttr("Z")
        capsule.CreateRadiusAttr(radius)
        end_shift = 0.5 * radius if i == 0 else -0.5 * radius if i == num_segments - 1 else 0.0
        capsule.CreateHeightAttr(segment_length - abs(2 * end_shift))  # cylinder part
        if end_shift:
            capsule.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, end_shift))
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())

        if i == 0:  # clamp: fixed joint from the world to the bottom end of segment 0
            joint = UsdPhysics.FixedJoint.Define(stage, "/Stem/joints/base")
            joint.CreateBody1Rel().SetTargets([segment.GetPath()])
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -0.5 * segment_length))
            continue
        joint = UsdPhysics.Joint.Define(stage, f"/Stem/joints/joint_{i:02d}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/Stem/seg_{i - 1:02d}")])
        joint.CreateBody1Rel().SetTargets([segment.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5 * segment_length))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -0.5 * segment_length))
        joint.CreateLocalRot0Attr(joint_rot)
        joint.CreateLocalRot1Attr(joint_rot)
        for axis in ("transX", "transY", "transZ"):  # low > high: locked
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)
        for axis in ("rotX", "rotY", "rotZ"):  # spring drives towards 0; gains come from the actuator cfg
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
            drive.CreateTypeAttr("force")
            drive.CreateTargetPositionAttr(0.0)
    # write only on change: Kit watches open layers and asks to reload a file that was rewritten
    content = stage.GetRootLayer().ExportToString()
    if not os.path.isfile(path) or open(path).read() != content:
        with open(path, "w") as file:
            file.write(content)


def spawn_chain(prim_path: str, cfg: "ChainUsdFileCfg", *args, **kwargs):
    """Write the chain USD from `cfg`, then spawn it like any USD file."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd  # needs the running app

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    write_chain_usd(cfg.usd_path, cfg.num_segments, cfg.length, cfg.diameter, cfg.density)
    return spawn_from_usd(prim_path, cfg, *args, **kwargs)


@configclass
class ChainUsdFileCfg(sim_utils.UsdFileCfg):
    """USD file spawner that first writes the chain USD from the geometry fields below."""

    func: Callable = spawn_chain

    num_segments: int = 20
    length: float = 0.4
    diameter: float = 0.008
    density: float = 1000.0


def stem_cfg() -> ArticulationCfg:
    """Vertical stem with its base at `geometry.base_position` in the env frame. Set `prim_path` in the scene."""
    params = stem_params("chain")
    geometry, material, joint, solver = params["geometry"], params["material"], params["joint"], params["solver"]
    usd_values = {key: geometry[key] for key in ("num_segments", "length", "diameter")} | {
        "density": material["density"]
    }
    usd_hash = hashlib.sha1(json.dumps(usd_values, sort_keys=True).encode()).hexdigest()[:8]

    segment_length = geometry["length"] / geometry["num_segments"]
    area_moment = math.pi * geometry["diameter"] ** 4 / 64
    bend_stiffness = material["bend_modulus"] * area_moment / segment_length
    twist_modulus = material["twist_modulus"]
    twist_stiffness = bend_stiffness if twist_modulus is None else twist_modulus * 2.0 * area_moment / segment_length
    damping_time = material["damping_time"] or 0.0

    def spring(joint_names: str, stiffness: float) -> ImplicitActuatorCfg:
        return ImplicitActuatorCfg(
            joint_names_expr=[joint_names],
            stiffness=stiffness,
            damping=damping_time * stiffness,
            armature=joint["armature"],
        )

    return ArticulationCfg(
        spawn=ChainUsdFileCfg(
            usd_path=str(BUILD_DIR / f"chain_{usd_hash}.usda"),
            activate_contact_sensors=True,  # for a ContactSensor on the segments
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=solver["position_iterations"],
                    solver_velocity_iteration_count=solver["velocity_iterations"],
                )
            ],
            **usd_values,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(geometry["base_position"])),
        actuators={"bend": spring(r"joint_\d+:[12]", bend_stiffness), "twist": spring(r"joint_\d+:0", twist_stiffness)},
    )


def physics_cfg() -> PhysxCfg:
    """PhysX physics cfg (the step `solver.sim_dt` goes into the `SimulationCfg`, iterations into `stem_cfg()`)."""
    return PhysxCfg()


def fix_stem_base(stem: Articulation) -> None:
    """Nothing to do: the base is clamped by the fixed joint in the USD."""


def segment_poses(stem: Articulation) -> torch.Tensor:
    """Segment poses (num_envs, num_segments, 7): position + quaternion (x, y, z, w), world frame."""
    segment_ids, _ = stem.find_bodies("seg_.*")
    return stem.data.body_link_pose_w.torch[:, segment_ids]


def segment_masses(stem: Articulation) -> torch.Tensor:
    """Segment masses (num_envs, num_segments) [kg]."""
    segment_ids, _ = stem.find_bodies("seg_.*")
    return stem.data.body_mass.torch[:, segment_ids]


def joint_gains(stem: Articulation) -> dict[str, np.ndarray]:
    """Per-DOF stiffness [N m/rad], damping [N m s/rad] and armature [kg m^2] in the simulation."""
    data = stem.data
    return {
        "stiffness": data.joint_stiffness.torch.cpu().numpy(),
        "damping": data.joint_damping.torch.cpu().numpy(),
        "armature": data.joint_armature.torch.cpu().numpy(),
    }


def write_kick(stem: Articulation, angular_velocity: float) -> None:
    """Rotate the stem above the first joint rigidly about the world y axis (+x sideways at the top) [rad/s]."""
    velocity = torch.zeros_like(stem.data.joint_vel.torch)
    velocity[:, stem.find_joints("joint_01:1")[0]] = angular_velocity
    stem.write_joint_velocity_to_sim_index(velocity=velocity)


def segment_force_setter(stem: Articulation) -> Callable[[int, torch.Tensor | None], None]:
    """Returns `set_force(segment, force)`: constant force (num_envs, 3) [N, world frame] at the segment's centre
    of mass, replacing any previous one; `None` removes it. Applied from the next `scene.write_data_to_sim()`."""
    segment_ids, _ = stem.find_bodies("seg_.*")

    def set_force(segment: int, force: torch.Tensor | None) -> None:
        stem.permanent_wrench_composer.reset()
        if force is not None:
            stem.permanent_wrench_composer.set_forces_and_torques_index(
                forces=force.unsqueeze(1), body_ids=[segment_ids[segment]], is_global=True
            )

    return set_force
