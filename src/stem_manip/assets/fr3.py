"""Isaac Lab articulation config for the FR3 with any end-effector from assets/fr3/end_effectors/.

Uses the USD produced by assets/fr3/convert_to_usd.sh (assets/fr3/build/fr3_<ee>_usd/fr3_<ee>/fr3_<ee>.usda).
Joint position limits come from the URDF (FR3 values from franka_description).
Gravity is off for the robot (its links and the end-effector only; other assets keep theirs), as in Isaac Lab's
`FRANKA_PANDA_HIGH_PD_CFG` for task-space control: with gravity, differential IK on top of the PD drives keeps the
gravity sag as a steady error (about 4 cm at the tool tip). The real FR3 compensates gravity itself (libfranka
torque commands are sent without gravity and friction); its compensation includes the end-effector only if the
end-effector's mass, centre of mass and inertia are set as the robot's load (`setLoad` or a Desk profile).
Joint armature: the motors' reflected inertia (gear ratio^2 x motor inertia, from the `<dynamics>` tags of the URDF);
without it the simulated joints are far lighter than the real ones (docs/notes/2026-10-06.md).
Two control modes: "position" (joint PD with Isaac Lab's FRANKA_PANDA_HIGH_PD_CFG gains, for scripted checks with
differential IK; not the real robot's behaviour) and "torque" (no joint PD; the task's action term computes the
torques with Franka's Cartesian impedance law, as on the real robot).
Written against the Isaac Lab 3.0 API (backend-specific schemas, joint_effort_limit).
"""

import math
import xml.etree.ElementTree as ET

import yaml

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from stem_manip.assets import REPO_ASSETS_DIR

BUILD_DIR = REPO_ASSETS_DIR / "fr3" / "build"
EE_DIR = REPO_ASSETS_DIR / "fr3" / "end_effectors"

# Bodies for tasks: the end-effector body has the end-effector's name (e.g. "fork_v2").
# tool_tip and fr3_link8 are plain frames in the USD, not bodies (the converter turns massless links
# without geometry into frames). Use the end-effector body + tool_tip_offset(ee) instead.


def tool_tip_offset(ee: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Pose of tool_tip in the end-effector body `ee`, read from end_effectors/<ee>/<ee>.yaml.

    Returns (pos, rot) with rot as quaternion (x, y, z, w), the Isaac Lab 3.0 convention.
    Pass it as OffsetCfg(pos=pos, rot=rot) with body name `ee`, e.g. to the IK action or a FrameTransformer.
    """
    tip = yaml.safe_load((EE_DIR / ee / f"{ee}.yaml").read_text())["tool_tip"]
    # URDF rpy: fixed-axis roll, pitch, yaw -> R = Rz(yaw) Ry(pitch) Rx(roll)
    cr, sr = math.cos(tip["rpy"][0] / 2), math.sin(tip["rpy"][0] / 2)
    cp, sp = math.cos(tip["rpy"][1] / 2), math.sin(tip["rpy"][1] / 2)
    cy, sy = math.cos(tip["rpy"][2] / 2), math.sin(tip["rpy"][2] / 2)
    rot = (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
    return tuple(float(v) for v in tip["xyz"]), rot


def motor_armature(ee: str) -> dict[str, float]:
    """Reflected motor inertia per arm joint [kg m^2] = gear_ratio^2 x motor_inertia, from the URDF's `<dynamics>`."""
    root = ET.parse(BUILD_DIR / f"fr3_{ee}.urdf").getroot()
    armature = {}
    for joint in root.iter("joint"):
        dynamics = joint.find("dynamics")
        if joint.get("type") == "revolute" and dynamics is not None and "motor_inertia" in dynamics.attrib:
            armature[joint.get("name")] = float(dynamics.get("gear_ratio")) ** 2 * float(dynamics.get("motor_inertia"))
    return armature


def fr3_cfg(ee: str, control: str = "position") -> ArticulationCfg:
    """FR3 + end-effector `ee` (e.g. "fork_v2").

    control: "position" (joint PD, stiff gains, for differential IK in checks) or "torque" (joint PD off; the
    action term sets joint efforts, e.g. Franka's Cartesian impedance law).
    """
    if control not in ("position", "torque"):
        raise ValueError(f"Unknown control mode '{control}', expected 'position' or 'torque'")
    stiffness, damping = (400.0, 80.0) if control == "position" else (0.0, 0.0)
    armature = motor_armature(ee)
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BUILD_DIR / f"fr3_{ee}_usd" / f"fr3_{ee}" / f"fr3_{ee}.usda"),
            activate_contact_sensors=True,  # needed if you put a ContactSensor on the end-effector
            rigid_props=PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "fr3_joint1": 0.0,
                "fr3_joint2": -0.569,
                "fr3_joint3": 0.0,
                "fr3_joint4": -2.810,
                "fr3_joint5": 0.0,
                "fr3_joint6": 3.037,
                "fr3_joint7": 0.741,
            },
        ),
        actuators={
            "fr3_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-4]"],
                joint_effort_limit=87.0,
                stiffness=stiffness,
                damping=damping,
                armature={name: value for name, value in armature.items() if name[-1] in "1234"},
            ),
            "fr3_forearm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                joint_effort_limit=12.0,
                stiffness=stiffness,
                damping=damping,
                armature={name: value for name, value in armature.items() if name[-1] in "567"},
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
