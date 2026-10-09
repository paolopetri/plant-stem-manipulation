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
    desired position step = clip(a[:3], -1, 1) * max_step (length limited to max_step: speed cap); its change from
    the previous step is limited to max_step_change (acceleration limit); target position += step; the target is then
    kept within max_target_offset of the measured tool tip. The same for the rotation (rotation vector, angles). Per
    physics step: joint torques from `stem_manip.utils.impedance.cartesian_impedance_torque`. The orientation is
    free; at reset the targets start at the tool's pose (the start pose holds the fork horizontal).
    Design (user, 2026-10-07, option iii of docs/notes/2026-10-07.md): speed caps that let the policy approach and
    turn within an episode; small tracking error through a limit on speed *changes* (with velocity feedforward the
    lag comes from changes of speed, not from speed; action spaces that keep the tracking error small transfer
    better, Aljalbout et al., RA-L 2024); a hard force bound when the tool is blocked. Values (option B1, user
    2026-10-07/08) and their derivation: docs/overleaf_folder/open_questions/action_limits_problem.tex
    (Section "sec:al"). The robot must use `fr3_cfg(ee, control="torque")`.
    """

    class_type: type | str = "stem_manip.tasks.push_position.mdp.actions:ToolTipImpedanceAction"

    joint_names: list[str] = MISSING
    """Arm joints driven by the controller."""
    body_name: str = MISSING
    """Body that carries the tool (e.g. "fork_v2")."""
    tool_offset: tuple[tuple[float, float, float], tuple[float, float, float, float]] = MISSING
    """Tool tip in the body frame: (position, quaternion (x, y, z, w)), e.g. `tool_tip_offset("fork_v2")`."""
    max_step: float = 0.0032
    """Largest tool-tip translation per policy step [m] (3.2 mm at 31.25 Hz = 10 cm/s; user, 2026-10-08): reached
    and left again within ~13 cm at the acceleration limit (20 cm/s, 2026-10-07, needed 42 cm, more than the stem
    area, and lagged up to 3.4 mm over such moves; `scripts/sweep_action_poses.py --full_speed`)."""
    max_rot_step: float = 0.02513
    """Largest tool rotation per policy step [rad] (1.44 deg at 31.25 Hz = 45 deg/s; user, 2026-10-07)."""
    max_step_change: float = 0.00008
    """Largest change of the translation step per policy step [m] (0.08 mm = 0.08 m/s^2; acceleration limit, user
    2026-10-08): under constant acceleration the lag settles at Lambda a / K <= 1.72 mm (Lambda <= 21.5 kg), below the
    2 mm criterion (lag criterion after Aljalbout et al.); 0.1 m/s^2 (2026-10-07) reached 2.15 mm in real reversals.
    Overshoot <= 5 mm and fork tilt <= 1 deg hold too (`scripts/sweep_action_poses.py`, both modes). This holds up to
    x 0.60 m; at the far edge (x 0.75 m) Lambda grows to 40-76 kg and the lag to ~4 mm (known limit until the inertia
    feedforward B3, user 2026-10-09)."""
    max_rot_step_change: float = 0.000349
    """Largest change of the rotation step per policy step [rad] (0.02 deg = 20 deg/s^2, user 2026-10-07): angle
    following and overshoot <= 2 deg, tool-tip drift <= 2 mm (write-up Table "tab:al-b1")."""
    max_target_offset: float = 0.004
    """Largest distance of the target from the measured tool tip [m]; bounds the spring force to K_p * 4 mm = 4 N
    when the tool is blocked (user, 2026-10-08: 2 x the lag criterion, ~3 x the measured push forces 0.8-1.4 N)."""
    max_target_rot_offset: float = 0.05236
    """Largest angle between target and tool orientation [rad] (3 deg, user 2026-10-08): bounds the moment to
    K_o * 0.052 = 5.2 N m, 43 % of the FR3 wrist torque limit 12 N m (10 deg would give 17 N m, above it)."""
    stiffness_pos: float = 1000.0
    """Translational stiffness K_p [N/m] (option B1, user 2026-10-07; Franka's examples use <= 400, its internal
    Cartesian impedance accepts up to 3000; to be confirmed with the lab)."""
    stiffness_rot: float = 100.0
    """Rotational stiffness K_o [N m/rad] (option B1): fork tilt <= 1 deg at the B1 acceleration limit (write-up
    Table "tab:al-diag")."""
    stiffness_nullspace: float = 0.5
    """Null-space posture stiffness [N m/rad] (franka_ros default 0.5); rest posture = the robot's default joints."""
    damping: str = "apparent_mass"
    """Task-space damping: "franka" (2 sqrt(K), Franka's examples; critical only for a 1 kg tool) or "apparent_mass"
    (2 sqrt(K) Lambda^(1/2) with the inertia Lambda the tool feels, critical in every pose and direction; uses the
    mass matrix incl. joint armature, see `stem_manip.utils.impedance.apparent_mass_damping`)."""
    torque_rate_limit: float = 1000.0
    """Largest change of each joint torque [N m/s] (Franka: 1 N m per 1 ms)."""
