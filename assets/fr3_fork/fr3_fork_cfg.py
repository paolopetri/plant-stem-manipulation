"""Isaac Lab articulation config for the FR3 with the fork end-effector.

Uses the USD produced by convert_to_usd.sh (build/fr3_fork.usd).
Joint position limits come from the URDF (FR3 values from franka_description).
Written against the Isaac Lab 3.0 API (backend-specific schemas, joint_effort_limit).
"""

from pathlib import Path

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

USD_PATH = Path(__file__).resolve().parent / "build" / "fr3_fork.usd"

# Body names to use in tasks (observations, rewards, IK).
EE_BODY = "tool_tip"
FLANGE_BODY = "fr3_link8"

FR3_FORK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,  # needed if you put a ContactSensor on the fork
        rigid_props=PhysxRigidBodyCfg(disable_gravity=False, max_depenetration_velocity=5.0),
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
            stiffness=400.0,
            damping=80.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            joint_effort_limit=12.0,
            stiffness=400.0,
            damping=80.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""FR3 + fork, stiff PD gains (suited for task-space control)."""
