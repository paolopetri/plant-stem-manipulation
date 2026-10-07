"""Cfg of the tool-tip impedance action (`actions.ToolTipImpedanceAction`).

Kept apart from the action class, which imports Isaac Lab's asset classes and with them USD modules that must not be
loaded before the app starts; the cfg refers to the class by name (as Isaac Lab's own action cfgs do).
"""

from dataclasses import MISSING

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass


@configclass
class ToolTipImpedanceActionCfg(ActionTermCfg):
    """Relative tool-tip pose action (6-D), executed by Franka's Cartesian impedance law (as on the real FR3).

    Action = (position step, rotation step), both in the robot base frame (user, 2026-10-07). Per policy step:
    target position += clip(a[:3], -1, 1) * max_step (length limited to max_step); target orientation is turned by
    the rotation vector clip(a[3:], -1, 1) * max_rot_step (angle limited to max_rot_step). Per physics step: joint
    torques from `stem_manip.utils.impedance.cartesian_impedance_torque`. The orientation is free; at reset the
    targets start at the tool's pose (the start pose holds the fork horizontal).
    Values and their source: docs/overleaf_folder/open_questions/impedance_action_study.tex (2026-10-07): with
    apparent-mass damping, K_p 400 N/m, K_o 30 N m/rad and feedforward, steps up to 1.75 mm per policy step keep the
    following error <= 2 mm, the overshoot <= 5 mm and the fork's tilt <= 1 deg at 7 of 9 points of the workspace
    (`scripts/sweep_action_poses.py`; the other two are within 0.2 rad of joint 4's limit). The robot must use `fr3_cfg(ee, control="torque")`.
    """

    class_type: type | str = "stem_manip.tasks.push_position.mdp.actions:ToolTipImpedanceAction"

    joint_names: list[str] = MISSING
    """Arm joints driven by the controller."""
    body_name: str = MISSING
    """Body that carries the tool (e.g. "fork_v2")."""
    tool_offset: tuple[tuple[float, float, float], tuple[float, float, float, float]] = MISSING
    """Tool tip in the body frame: (position, quaternion (x, y, z, w)), e.g. `tool_tip_offset("fork_v2")`."""
    max_step: float = 0.00175
    """Largest tool-tip translation per policy step [m] (1.75 mm at 31.25 Hz = 5.5 cm/s)."""
    max_rot_step: float = 0.004363
    """Largest tool rotation per policy step [rad] (0.25 deg at 31.25 Hz = 7.8 deg/s). Largest step with angle following
    <= 2 deg, overshoot <= 2 deg and tool-tip drift <= 2 mm while turning, over the workspace (criterion: user,
    2026-10-07; `scripts/sweep_action_poses.py`: 0.30 deg drifts 2.07 mm). Slow on purpose: faster rotations need the
    rotation decoupled from the translation (planned after the first sim-to-real transfer, docs/TODO.md)."""
    stiffness_pos: float = 400.0
    """Translational stiffness K_p [N/m] (franka_ros example: up to 400)."""
    stiffness_rot: float = 30.0
    """Rotational stiffness K_o [N m/rad] (Franka examples: 10, franka_ros up to 30); with 10 the fork tilts > 1 deg
    at 2 mm steps."""
    stiffness_nullspace: float = 0.5
    """Null-space posture stiffness [N m/rad] (franka_ros default 0.5); rest posture = the robot's default joints."""
    damping: str = "apparent_mass"
    """Task-space damping: "franka" (2 sqrt(K), Franka's examples; critical only for a 1 kg tool) or "apparent_mass"
    (2 sqrt(K) Lambda^(1/2) with the inertia Lambda the tool feels, critical in every pose and direction; uses the
    mass matrix incl. joint armature, see `stem_manip.utils.impedance.apparent_mass_damping`)."""
    feedforward: bool = True
    """Move the target linearly through the policy step and feed its velocity forward (otherwise held)."""
    torque_rate_limit: float = 1000.0
    """Largest change of each joint torque [N m/s] (Franka: 1 N m per 1 ms)."""
