"""Env check for stage 1 (`StemManip-Push-Position-FR3-v0`): the env steps, and the fork follows the actions.

Creates the env with a few envs, resets it and drives the slot base with constant actions (same in every env):
hold (zero action), then full translation along +x, +y, +z, each for `--steps` policy steps and followed by a hold;
then a full rotation about the vertical (+z) for `--steps` policy steps and a hold. Then, each after a reset to the
start pose: a full +y translation and a full rotation about +z from rest (step limits), and a move +x followed by a
full move down (-z) into the ground and a lift (setpoint clamp, restart after contact); the clamp test starts from
the held-out start pose a_above_tip without joint noise (every env alike, as `check_push_terms.py`), so that the
ground contact does not vary with the random table pose. The action term (6-D:
translation and rotation step, robot base frame) is Franka's Cartesian impedance law with an
integrated setpoint, speed cap, step-change limit and setpoint-offset clamp (`mdp.SlotBaseImpedanceAction`). With the
step-change limit the setpoint ramps up at the start of a move and brakes during the hold.
The controller is tested without the stem: the spawn area is switched off and the stem stands out of the fork's reach
behind the robot (`STEM_AWAY`); the clamp test pushes the fork into the ground instead.
Checks:
- shapes: observation (num_envs, 57) and action (num_envs, 6), all observations finite;
- start pose: at the first reset the arm joints of every env within 0.02 rad (the reset noise, proposal 2026-10-15) of
  a pose of the start-pose table (`env_cfg.START_POSES`, checked in `tests/test_fr3_reference.py`); the slot base
  within 1 deg of its orientation at reset after the hold;
- null-space rest posture: at every reset, the action term's null-space rest posture equals the arm joints measured
  right after the reset (the start pose picked, with its noise; user, 2026-10-15, #163, as Franka's example
  controller), to 1e-6 rad;
- holds still: the slot base moves less than 1 mm during the last half of the hold;
- follows the actions: following error (largest distance to the setpoint over the move and the hold; at the end of
  a policy step the interpolated setpoint equals the step's end setpoint) <= 2 mm; the tool passes the setpoint's final
  position by at most 5 mm and ends within 1 mm of it; sideways drift < 2 mm; orientation within 1 deg throughout.
  The final setpoint is the setpoint after the clamp; that every commanded step was executed follows from the following
  error: had the clamp shortened a step, setpoint and tool would be 4 mm apart at the end of that policy step;
- follows the rotation: about +z, angle following <= 2 deg, overshoot past the final setpoint <= 2 deg, slot-base
  drift while turning <= 2 mm, end within 1 deg of the setpoint after the hold;
- step limits: from rest the first step is exactly 0.08 mm (+y translation) and 0.02 deg (rotation about +z), and
  the step then reaches the caps 3.2 mm and 1.44 deg;
- follows at full speed: over the +y ramp (incl. `FULL_SPEED_STEPS` steps at the cap, 10 cm/s) following error
  <= 2 mm; over the rotation ramp (up to 45 deg/s) angle following <= 2 deg and slot-base drift <= 2 mm (the moves
  above stay below the caps because of the step-change limit);
- setpoint clamp: at every physics step of the move down, the setpoint the law uses is at most 4 mm from the tool; while
  the tool is blocked by the ground (last `BLOCKED_STEPS` policy steps) exactly 4 mm;
- contact force at the ground: a contact sensor on the fork (net force, check only; the task's sensor is on the
  stem). While blocked (last `BLOCKED_STEPS` policy steps), its mean over each policy step is > 0 and <= 4.2 N (the
  clamp's spring bound 4 mm x K_p 1000 = 4 N, + 5 %) and varies by at most 0.2 N (no chatter); the impact before is
  reported, not bounded (user, 2026-10-15; #77);
- restart after contact: lifting off the ground, the applied step changes by at most 0.08 mm per policy step;
- applied step within the caps (3.2 mm, 1.44 deg) in every phase;
- slot-base observations = independent kinematics, after every policy step of every phase (incl. the clamp): the
  observed slot-base position and orientation (first two columns of its rotation matrix) equal the slot base from the
  measured joint angles via the FR3 Denavit-Hartenberg parameters and the yaml's slot_base
  (`stem_manip.utils.fr3_reference`, no code shared with the observations); the observed applied step x the decided
  caps equals the step the decided limits (cap, step-change limit) give for the action, computed here (independent,
  but only in the policy steps where the clamp did not act; the row reports n of N), and the setpoint's motion over the
  policy step (self-consistency only: the setpoint is the action term's own state); the observed setpoint offset x the
  decided bounds (4 mm, 3 deg) equals the setpoint minus that slot base. Tolerance 1e-5 m and 1e-5 rad (user,
  2026-10-14): both sides use the same nominal kinematics, so only float32 rounding remains (about 1e-6 over the
  chain); 10x margin.
The expected limits are the decided values (option B1), not the cfg's: with an override flag those checks fail.
Joint-limit guard: a check whose measurement window comes within 0.25 rad of a joint limit fails as invalid (there
the arm, not the controller, limits the motion; user 2026-10-08, same threshold as scripts/sweep_action_poses.py).
The orientation reference for translation moves is the start orientation (rotation action zero).
Reported: per axis, travel, largest following error and overshoot.

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/check_push_env.py
    uv run --extra isaacsim python scripts/check_push_env.py --num_envs 1 --viz kit --slow_motion 3
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Env check for the stage-1 push task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs.")
parser.add_argument("--steps", type=int, default=30, help="Policy steps per move.")
parser.add_argument("--hold_steps", type=int, default=45, help="Policy steps of zero action after each move (1.44 s).")
parser.add_argument(
    "--damping", choices=["franka", "apparent_mass"], default=None, help="Override the action's damping (cfg default)."
)
parser.add_argument("--rot_stiffness", type=float, default=None, help="Override the action's K_o [N m/rad].")
parser.add_argument("--max_step", type=float, default=None, help="Override the action's max_step [m] (cfg default).")
parser.add_argument("--max_rot_step", type=float, default=None, help="Override the action's max_rot_step [rad].")
parser.add_argument(
    "--slow_motion", type=float, default=1.0, help="With a viewer: play this many times slower than real time."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import math
import time

import gymnasium as gym
import torch

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat

import stem_manip.tasks  # noqa: F401  (registers the task)
from stem_manip.tasks.push_position.env_cfg import START_POSES, StemPushPositionEnvCfg, fixed_start
from stem_manip.utils import check_report
from stem_manip.utils.fr3_reference import rotvec_from_matrix, slot_base_pose
from stem_manip.utils.impedance import rotvec_between

TASK = "StemManip-Push-Position-FR3-v0"
STEM_AWAY = (-0.8, 0.8, 0.0)  # [m] stem base, out of the fork's reach behind the robot (as in sweep_action_poses.py)
JOINT_NOISE = 0.02  # [rad] uniform reset noise per arm joint (user, 2026-10-15, #163)
NOISE_TOL = 1e-5  # [rad] float32 rounding of the joint angles
POSTURE_TOL = 1e-6  # [rad] null-space rest posture vs the measured joints at reset (a copy: rounding only)
ORIENTATION_TOL = math.radians(1.0)
HOLD_TOL = 1e-3  # [m]
FOLLOW_TOL = 2e-3  # [m] following error while moving
OVERSHOOT_TOL = 5e-3  # [m]
END_TOL = 1e-3  # [m] after the hold following a move
DRIFT_TOL = 2e-3  # [m]
ROT_FOLLOW_TOL, ROT_OVERSHOOT_TOL, ROT_DRIFT_TOL = math.radians(2.0), math.radians(2.0), 2e-3  # user, 2026-10-07
OBS_DIM = 57  # slot base: position 3, orientation 6, applied step 6, setpoint offset 6; stem base 3, 5 stem points x 2
OBS_SLOT_BASE_POS, OBS_SLOT_BASE_ROT = (
    slice(0, 3),
    slice(3, 9),
)  # decided layout (user, 2026-10-08), not read from the cfg
OBS_STEP_POS, OBS_STEP_ROT, OBS_OFFSET_POS, OBS_OFFSET_ROT = slice(9, 12), slice(12, 15), slice(15, 18), slice(18, 21)
# slot-base observation vs independent value: rows of the check; the ROT ones in rad, the others in m
SLOT_BASE_ERRORS = (
    "position",
    "orientation",
    "applied step = setpoint motion (self-consistency)",
    "applied rotation = setpoint turn (self-consistency)",
    "applied step = limits (independent)",
    "applied rotation = limits (independent)",
    "setpoint offset",
    "setpoint rotation offset",
)
ROT_ERRORS = (
    "orientation",
    "applied rotation = setpoint turn (self-consistency)",
    "applied rotation = limits (independent)",
    "setpoint rotation offset",
)
LIMITS_ERRORS = (
    "applied step = limits (independent)",
    "applied rotation = limits (independent)",
)  # compared only where the clamp did not act
# decided action limits (option B1, user 2026-10-07/08; docs/overleaf_folder/open_questions/action_limits_problem.tex);
# fixed here, not read from the cfg, so that a wrong cfg value fails (with an override flag these checks fail too)
FIRST_STEP, FIRST_ROT_STEP = 8e-5, 0.000349  # [m], [rad] step-change limits 0.08 mm (2026-10-08) / 0.02 deg
CAP_STEP, CAP_ROT_STEP = 0.0032, 0.02513  # [m], [rad] speed caps 10 cm/s (2026-10-08) / 45 deg/s (2026-10-07)
CLAMP_OFFSET = 4e-3  # [m] setpoint clamp, 4 N at K_p 1000 (2026-10-08)
# [N] blocked: policy-step mean <= spring bound 4 mm x K_p 1000 = 4 N, + 5 % (user, 2026-10-15)
CLAMP_FORCE = 4.2
CHATTER_TOL = 0.2  # [N] largest spread of the policy-step mean while blocked (user, 2026-10-15)
CLAMP_ROT_OFFSET = 0.05236  # [rad] orientation clamp, 3 deg (2026-10-08)
SLOT_BASE_POS_TOL, SLOT_BASE_ROT_TOL = (
    1e-5,
    1e-5,
)  # [m], [rad] slot-base obs vs independent kinematics (user, 2026-10-14)
JOINT_MARGIN_MIN = 0.25  # [rad] closer to a joint limit, a check is invalid (user, 2026-10-08)
# clamp test path (test setup, not a criterion): +x first, so that the way down stays clear of joint 4's limit
# (DOWN_STEPS leaves the blocked window well after the tool has reached the ground)
CLAMP_APPROACH_STEPS, DOWN_STEPS, BLOCKED_STEPS, LIFT_STEPS = 40, 260, 20, 10
FULL_SPEED_STEPS = 10  # policy steps at the translation cap at the end of the +y ramp


def main() -> None:
    """Run the phases and print a pass/fail summary."""
    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.episode_length_s = 1e4  # no time-out reset in the middle of the phases
    # no joint-margin reset in the middle of a phase: the check guards the margin itself (JOINT_MARGIN_MIN)
    env_cfg.terminations.joint_margin = None
    env_cfg.events.spawn_stem = None  # the fork must not touch the stem: no spawn area, stem out of reach
    env_cfg.scene.stem.init_state.pos = STEM_AWAY
    # contact force of the fork at the ground (net force: the stem is out of reach), every physics step of a policy step
    env_cfg.scene.fork_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*fork_v2", history_length=env_cfg.decimation
    )
    if args_cli.rot_stiffness is not None:
        env_cfg.actions.slot_base.stiffness_rot = args_cli.rot_stiffness
    if args_cli.max_step is not None:
        env_cfg.actions.slot_base.max_step = args_cli.max_step
    if args_cli.max_rot_step is not None:
        env_cfg.actions.slot_base.max_rot_step = args_cli.max_rot_step
    if args_cli.damping is not None:
        env_cfg.actions.slot_base.damping = args_cli.damping
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        obs, _ = env.reset()
        sim, device, n = env.sim, env.device, env.num_envs
        term = env.action_manager.get_term("slot_base")
        robot = env.scene[term.cfg.asset_name]
        fk_joint_ids = robot.find_joints([f"fr3_joint{i}" for i in range(1, 8)], preserve_order=True)[0]
        q_reset = robot.data.joint_pos.torch[:, fk_joint_ids]
        table = torch.tensor(START_POSES, device=device)
        start_error = float((q_reset[:, None] - table[None]).abs().amax(dim=-1).min(dim=-1).values.max())
        quat_ref = term.slot_base_pose()[1].clone()  # start orientation, held while the rotation action is zero
        term_joint_ids = robot.find_joints(term.cfg.joint_names)[0]
        posture_errors: list[float] = []  # null-space rest posture vs the measured arm joints, at every reset [rad]

        def record_posture() -> None:
            q = robot.data.joint_pos.torch[:, term_joint_ids]
            posture_errors.append(float((term.nullspace_posture() - q).abs().max()))

        record_posture()

        def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Angle between orientations a and b (n, 4) [rad] (axis-angle of a b^-1; exact also near 0)."""
            return rotvec_between(a, b).norm(dim=-1)

        def angle_to_fixed(quat: torch.Tensor) -> torch.Tensor:
            return angle_between(quat, quat_ref)

        step_time = env.step_dt * args_cli.slow_motion
        frame_start = time.perf_counter()
        finite = bool(torch.isfinite(obs["policy"]).all())
        max_angle = 0.0
        joint_ids = robot.find_joints(term.cfg.joint_names)[0]
        limits = robot.data.joint_pos_limits.torch[:, joint_ids]  # (n, joints, 2)
        margin = math.inf  # smallest distance to a joint limit [rad] since the last `window()`
        max_applied = [0.0, 0.0]  # largest applied translation [m] and rotation [rad] step over all phases
        # largest difference of each slot-base observation from its independent value [m or rad]
        obs_errors = dict.fromkeys(SLOT_BASE_ERRORS, 0.0)
        obs_checked, limits_checked = 0, 0  # policy steps compared (limits: steps where the clamp did not act)
        max_setpoint_offset = (
            0.0  # largest observed setpoint offset (position part, normalized), to show the clamp's range
        )
        expected_step = torch.zeros(n, 6, device=device)  # applied step [m, rad] expected from the action and limits

        def limited(action: torch.Tensor, prev: torch.Tensor, cap: float, change: float) -> torch.Tensor:
            """Step from the decided limits (option B1): clip(action, -1, 1) x cap, length <= cap; its change from
            the previous applied step `prev` of length <= change; the result again of length <= cap."""

            def limit(v: torch.Tensor, bound: float) -> torch.Tensor:
                return v * (bound / v.norm(dim=-1, keepdim=True).clamp(min=bound))

            wanted = limit(action.clamp(-1.0, 1.0) * cap, cap)
            return limit(prev + limit(wanted - prev, change), cap)

        def slot_base_obs_errors(
            policy: torch.Tensor,
            setpoint_before: torch.Tensor,
            setpoint_quat_before: torch.Tensor,
            action: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            """Differences (n,) of the slot-base slices of `policy` from the independent kinematics, from the setpoint's
            motion since (`setpoint_before`, `setpoint_quat_before`) and from the step the limits give for `action`
            (`SLOT_BASE_ERRORS`, see the module docstring). Then sets `expected_step` to the step actually applied (the
            setpoint's motion, after the clamp), from which the next step's limits start."""
            slot_base = slot_base_pose(robot.data.joint_pos.torch[:, fk_joint_ids], term.cfg.body_name)
            slot_base_pos, slot_base_rot = slot_base[:, :3, 3], slot_base[:, :3, :3]
            obs = policy.double()
            setpoint = term.setpoint().double()
            setpoint_rot = matrix_from_quat(term.setpoint_quat()).double()
            turned = rotvec_from_matrix(setpoint_rot @ matrix_from_quat(setpoint_quat_before).double().transpose(1, 2))
            step_pos, step_rot = obs[:, OBS_STEP_POS] * CAP_STEP, obs[:, OBS_STEP_ROT] * CAP_ROT_STEP
            limits_pos = limited(action[:3].expand(n, 3), expected_step[:, :3], CAP_STEP, FIRST_STEP).double()
            limits_rot = limited(action[3:].expand(n, 3), expected_step[:, 3:], CAP_ROT_STEP, FIRST_ROT_STEP).double()
            slot_base_rot6d = torch.cat((slot_base_rot[..., 0], slot_base_rot[..., 1]), dim=-1)
            offset_rot = rotvec_from_matrix(setpoint_rot @ slot_base_rot.transpose(1, 2))
            errors = {
                "position": (obs[:, OBS_SLOT_BASE_POS] - slot_base_pos).norm(dim=-1),
                "orientation": (obs[:, OBS_SLOT_BASE_ROT] - slot_base_rot6d).abs().amax(dim=-1),
                "applied step = setpoint motion (self-consistency)": (
                    step_pos - (setpoint - setpoint_before.double())
                ).norm(dim=-1),
                "applied rotation = setpoint turn (self-consistency)": (step_rot - turned).norm(dim=-1),
                "applied step = limits (independent)": (step_pos - limits_pos).norm(dim=-1),
                "applied rotation = limits (independent)": (step_rot - limits_rot).norm(dim=-1),
                "setpoint offset": (obs[:, OBS_OFFSET_POS] * CLAMP_OFFSET - (setpoint - slot_base_pos)).norm(dim=-1),
                "setpoint rotation offset": (obs[:, OBS_OFFSET_ROT] * CLAMP_ROT_OFFSET - offset_rot).norm(dim=-1),
            }
            expected_step[:] = torch.cat((setpoint - setpoint_before.double(), turned), dim=-1).to(expected_step.dtype)
            return errors

        def window() -> None:
            """Start a measurement window of the joint-limit guard."""
            nonlocal margin
            margin = math.inf

        def step(action: torch.Tensor, track_tilt: bool = True) -> None:
            nonlocal frame_start, finite, max_angle, margin, obs_checked, limits_checked, max_setpoint_offset
            setpoint_before, setpoint_quat_before = term.setpoint().clone(), term.setpoint_quat().clone()
            obs, _, _, _, _ = env.step(action.repeat(n, 1))
            finite = finite and bool(torch.isfinite(obs["policy"]).all())
            errors = slot_base_obs_errors(obs["policy"], setpoint_before, setpoint_quat_before, action)
            free = ~term.clamp_held().any(dim=-1)  # where the clamp acted, the limits' step is not the applied one
            for name, error in errors.items():
                error = error[free] if name in LIMITS_ERRORS else error
                if error.numel() > 0:
                    obs_errors[name] = max(obs_errors[name], float(error.max()))
            obs_checked += 1
            limits_checked += int(free.all())
            max_setpoint_offset = max(max_setpoint_offset, float(obs["policy"][:, OBS_OFFSET_POS].norm(dim=-1).max()))
            q = robot.data.joint_pos.torch[:, joint_ids]
            margin = min(margin, float(torch.minimum(q - limits[..., 0], limits[..., 1] - q).min()))
            applied = term.processed_actions
            max_applied[0] = max(max_applied[0], float(applied[:, :3].norm(dim=-1).max()))
            max_applied[1] = max(max_applied[1], float(applied[:, 3:].norm(dim=-1).max()))
            if track_tilt:
                max_angle = max(max_angle, float(angle_to_fixed(term.slot_base_pose()[1]).max()))
            if sim.is_rendering:
                time.sleep(max(0.0, step_time - (time.perf_counter() - frame_start)))
                frame_start = time.perf_counter()

        obs_shape, action_shape = tuple(obs["policy"].shape), tuple(env.action_manager.action.shape)
        pos0 = term.slot_base_pose()[0].clone()

        # -- hold: zero action
        zero = torch.zeros(6, device=device)
        for i in range(args_cli.steps):
            step(zero)
            if i == args_cli.steps // 2 - 1:
                pos_mid = term.slot_base_pose()[0].clone()
        pos_hold, quat_hold = (v.clone() for v in term.slot_base_pose())
        hold_move = float((pos_hold - pos_mid).norm(dim=-1).max())
        hold_angle = float(angle_to_fixed(quat_hold).max())

        # -- full action along +x, +y, +z, each followed by a hold
        moves = []
        window()
        for axis in range(3):
            action = torch.zeros(6, device=device)
            action[axis] = 1.0
            start = term.slot_base_pose()[0].clone()
            follow = torch.zeros(n, device=device)
            positions = []  # slot base positions during the hold
            for i in range(args_cli.steps + args_cli.hold_steps):
                step(action if i < args_cli.steps else zero)
                follow = torch.maximum(follow, (term.slot_base_pose()[0] - term.setpoint()).norm(dim=-1))
                if i >= args_cli.steps:
                    positions.append(term.slot_base_pose()[0][:, axis].clone())
            end = term.setpoint().clone()  # final setpoint (the setpoint brakes during the hold)
            overshoot = (torch.stack(positions) - end[:, axis]).max(dim=0).values
            pos = term.slot_base_pose()[0]
            delta = pos - start
            drift = torch.cat([delta[:, :axis], delta[:, axis + 1 :]], dim=1).norm(dim=-1)
            moves.append(
                {
                    "travel": delta[:, axis],
                    "follow": follow,
                    "overshoot": overshoot.clamp(min=0.0),
                    "end_error": (pos - end).norm(dim=-1),
                    "drift": drift,
                }
            )

        # -- rotation about the vertical (+z), checked as "follows the rotation" below
        rotate = torch.zeros(6, device=device)
        rotate[5] = 1.0
        moves_margin = margin
        window()
        slot_base_start = term.slot_base_pose()[0].clone()
        rot_follow, slot_base_drift = torch.zeros(n, device=device), torch.zeros(n, device=device)
        rot_speed = 0.0  # largest applied rotation step [rad]
        quats = []  # slot base orientations during the hold
        for i in range(args_cli.steps + args_cli.hold_steps):
            step(rotate if i < args_cli.steps else zero, track_tilt=False)
            slot_base_drift = torch.maximum(slot_base_drift, (term.slot_base_pose()[0] - slot_base_start).norm(dim=-1))
            rot_follow = torch.maximum(rot_follow, angle_between(term.slot_base_pose()[1], term.setpoint_quat()))
            rot_speed = max(rot_speed, float(term.processed_actions[:, 3:].norm(dim=-1).max()))
            if i >= args_cli.steps:
                quats.append(term.slot_base_pose()[1].clone())
        setpoint_end = term.setpoint_quat().clone()
        # overshoot: rotation from the final setpoint to the tool, about +z (the turning direction)
        rot_overshoot = torch.stack([rotvec_between(q, setpoint_end)[:, 2] for q in quats]).max(0).values
        rot_end = angle_between(term.slot_base_pose()[1], setpoint_end)
        turned = angle_between(setpoint_end, quat_ref)

        rot_margin = margin

        def restart() -> None:
            """Reset to the start pose and hold (zero action) until the tool settles."""
            env.reset()
            record_posture()
            expected_step.zero_()  # the reset starts the action term from rest
            for _ in range(args_cli.steps):
                step(zero, track_tilt=False)

        # -- step limits: from rest at the start pose, full +y translation, then full rotation about +z; the step
        # grows by the step-change limit until the cap
        def ramp(axis: int, n_steps: int) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, torch.Tensor]]:
            """Full action on `axis` from rest; returns the first and last applied step, the joint margin and the
            largest following error, angle following and slot-base drift over the ramp."""
            restart()
            window()
            action = torch.zeros(6, device=device)
            action[axis] = 1.0
            taken = []
            slot_base_start = term.slot_base_pose()[0].clone()
            follow = {k: torch.zeros(n, device=device) for k in ("pos", "angle", "drift")}
            for _ in range(n_steps):
                step(action, track_tilt=False)
                taken.append(term.processed_actions.clone())
                pos, quat = term.slot_base_pose()
                follow["pos"] = torch.maximum(follow["pos"], (pos - term.setpoint()).norm(dim=-1))
                follow["angle"] = torch.maximum(follow["angle"], angle_between(quat, term.setpoint_quat()))
                follow["drift"] = torch.maximum(follow["drift"], (pos - slot_base_start).norm(dim=-1))
            return taken[0], taken[-1], margin, follow

        ramp_steps = math.ceil(CAP_STEP / FIRST_STEP) + FULL_SPEED_STEPS
        rot_ramp_steps = math.ceil(CAP_ROT_STEP / FIRST_ROT_STEP) + 2
        first, last, ramp_margin, ramp_follow = ramp(1, ramp_steps)
        first_rot, last_rot, rot_ramp_margin, rot_ramp_follow = ramp(5, rot_ramp_steps)

        # -- setpoint clamp: +x, then full move down into the ground; at every physics step the setpoint the law uses
        # may be at most CLAMP_OFFSET ahead of the tool (measured around each apply_actions call); from a_above_tip
        # without joint noise (the ground contact must not vary with the random table pose)
        fixed_start(env.event_manager.get_term_cfg("start_pose").params)
        restart()
        x_move = torch.zeros(6, device=device)
        x_move[0] = 1.0
        for i in range(CLAMP_APPROACH_STEPS + args_cli.hold_steps):
            step(x_move if i < CLAMP_APPROACH_STEPS else zero, track_tilt=False)
        stretch = torch.zeros(
            n, device=device
        )  # largest setpoint-tool distance over the physics steps of a policy step
        apply_actions = term.apply_actions

        def apply_and_measure() -> None:
            pos = term.slot_base_pose()[0].clone()  # the pose the law sees in this physics step
            apply_actions()
            stretch.copy_(torch.maximum(stretch, (term.command_pose()[0] - pos).norm(dim=-1)))

        term.apply_actions = apply_and_measure
        fork_contact = env.scene["fork_contact"]
        down = torch.zeros(6, device=device)
        down[2] = -1.0
        stretches, ground_forces = [], []
        for i in range(DOWN_STEPS):
            if i == DOWN_STEPS - BLOCKED_STEPS:
                window()
            stretch.zero_()
            step(down, track_tilt=False)
            stretches.append(stretch.clone())
            # (n, physics steps of this policy step): fork net contact force [N]
            ground_forces.append(fork_contact.data.net_forces_w_history.torch.norm(dim=-1).amax(dim=-1))
        term.apply_actions = apply_actions
        stretches = torch.stack(stretches)  # (DOWN_STEPS, n)
        blocked = stretches[-BLOCKED_STEPS:]
        ground_forces = torch.stack(ground_forces)  # (DOWN_STEPS, n, decimation)
        blocked_means = ground_forces[-BLOCKED_STEPS:].mean(dim=-1)  # (BLOCKED_STEPS, n): mean over each policy step
        ground_spread = float((blocked_means.max(dim=0).values - blocked_means.min(dim=0).values).max())
        clamp_margin = margin
        slot_base_height = term.slot_base_pose()[0][:, 2]
        # -- restart after contact: lift; the applied step changes by at most the step-change limit
        prev = term.processed_actions[:, :3].clone()
        lift_change = torch.zeros(n, device=device)
        for _ in range(LIFT_STEPS):
            step(-down, track_tilt=False)
            lift_change = torch.maximum(lift_change, (term.processed_actions[:, :3] - prev).norm(dim=-1))
            prev = term.processed_actions[:, :3].clone()

        follows = all(
            bool((m["follow"] <= FOLLOW_TOL).all())
            and bool((m["overshoot"] <= OVERSHOOT_TOL).all())
            and bool((m["end_error"] <= END_TOL).all())
            and bool((m["drift"] <= DRIFT_TOL).all())
            for m in moves
        )
        results = {
            "shapes": (
                obs_shape == (n, OBS_DIM) and action_shape == (n, 6) and finite,
                f"observation {obs_shape}, action {action_shape}, all finite: {finite}",
            ),
            "start pose": (
                start_error <= JOINT_NOISE + NOISE_TOL and hold_angle < ORIENTATION_TOL,
                f"arm joints at reset {start_error:.4f} rad from a table pose, "
                f"orientation {math.degrees(hold_angle):.2f} deg (at reset: {[round(float(v), 3) for v in pos0[0]]} m)",
            ),
            "holds still": (
                hold_move < HOLD_TOL,
                f"slot base moved {hold_move * 1e3:.2f} mm in the 2nd half of the hold",
            ),
            "follows the actions": (
                follows and max_angle < ORIENTATION_TOL and moves_margin >= JOINT_MARGIN_MIN,
                "; ".join(
                    f"{'xyz'[a]}: travel {float(m['travel'].min()) * 1e3:.1f} mm, following "
                    f"{float(m['follow'].max()) * 1e3:.2f} mm, overshoot {float(m['overshoot'].max()) * 1e3:.2f} mm, "
                    f"end {float(m['end_error'].max()) * 1e3:.2f} mm, drift {float(m['drift'].max()) * 1e3:.2f} mm"
                    for a, m in enumerate(moves)
                )
                + f"; max orientation error {math.degrees(max_angle):.2f} deg; joint margin {moves_margin:.2f} rad",
            ),
            "null-space rest posture = start pose": (
                max(posture_errors) <= POSTURE_TOL,
                f"{len(posture_errors)} resets: largest difference from the arm joints at reset "
                f"{max(posture_errors):.1e} rad (tol {POSTURE_TOL:.0e})",
            ),
            "step limits": (
                bool(((first[:, :3].norm(dim=-1) - FIRST_STEP).abs() < 1e-6).all())
                and bool(((first_rot[:, 3:].norm(dim=-1) - FIRST_ROT_STEP).abs() < 1e-6).all())
                and bool(((last[:, :3].norm(dim=-1) - CAP_STEP).abs() < 1e-6).all())
                and bool(((last_rot[:, 3:].norm(dim=-1) - CAP_ROT_STEP).abs() < 1e-5).all())
                and min(ramp_margin, rot_ramp_margin) >= JOINT_MARGIN_MIN,
                f"+y from rest: first step {float(first[:, :3].norm(dim=-1).max()) * 1e3:.3f} mm (expected "
                f"{FIRST_STEP * 1e3:.3f}), after {ramp_steps} steps "
                f"{float(last[:, :3].norm(dim=-1).min()) * 1e3:.3f} mm (cap {CAP_STEP * 1e3:.3f}); about +z: first "
                f"{math.degrees(float(first_rot[:, 3:].norm(dim=-1).max())):.3f} deg (expected "
                f"{math.degrees(FIRST_ROT_STEP):.3f}), after {rot_ramp_steps} steps "
                f"{math.degrees(float(last_rot[:, 3:].norm(dim=-1).min())):.3f} deg "
                f"(cap {math.degrees(CAP_ROT_STEP):.3f}); joint margin {min(ramp_margin, rot_ramp_margin):.2f} rad",
            ),
            "follows at full speed": (
                float(ramp_follow["pos"].max()) <= FOLLOW_TOL
                and float(rot_ramp_follow["angle"].max()) <= ROT_FOLLOW_TOL
                and float(rot_ramp_follow["drift"].max()) <= ROT_DRIFT_TOL
                and min(ramp_margin, rot_ramp_margin) >= JOINT_MARGIN_MIN,
                f"+y ramp to {CAP_STEP / env.step_dt * 100:.0f} cm/s ({FULL_SPEED_STEPS} steps at the cap): following "
                f"{float(ramp_follow['pos'].max()) * 1e3:.2f} mm; rotation ramp to 45 deg/s: angle following "
                f"{math.degrees(float(rot_ramp_follow['angle'].max())):.2f} deg, slot-base drift "
                f"{float(rot_ramp_follow['drift'].max()) * 1e3:.2f} mm",
            ),
            "setpoint clamp": (
                bool((stretches <= CLAMP_OFFSET + 1e-6).all())
                and bool(((blocked - CLAMP_OFFSET).abs() < 1e-6).all())
                and clamp_margin >= JOINT_MARGIN_MIN,
                f"moving down into the ground: largest setpoint-tool distance {float(stretches.max()) * 1e3:.4f} mm "
                f"(bound {CLAMP_OFFSET * 1e3:.1f} mm), while blocked {float(blocked.min()) * 1e3:.4f}-"
                f"{float(blocked.max()) * 1e3:.4f} mm, slot base at z {float(slot_base_height.min()) * 1e3:.1f}-"
                f"{float(slot_base_height.max()) * 1e3:.1f} mm; joint margin {clamp_margin:.2f} rad",
            ),
            "contact force at the ground": (
                float(blocked_means.max()) <= CLAMP_FORCE
                and float(blocked_means.min()) > 0.0
                and ground_spread <= CHATTER_TOL,
                f"while blocked: mean over a policy step {float(blocked_means.min()):.3f}-"
                f"{float(blocked_means.max()):.3f} N (bound {CLAMP_FORCE} N), largest spread in one env "
                f"{ground_spread:.3f} N (tol {CHATTER_TOL:.2f} N); impact (not bounded): largest fork contact force "
                f"{float(ground_forces.max()):.3f} N at a physics step",
            ),
            "restart after contact": (
                bool((lift_change <= FIRST_STEP + 1e-6).all()),
                f"lifting off: largest change of the applied step {float(lift_change.max()) * 1e3:.4f} mm per step "
                f"(limit {FIRST_STEP * 1e3:.2f} mm)",
            ),
            "applied step within the caps": (
                max_applied[0] <= CAP_STEP + 1e-6 and max_applied[1] <= CAP_ROT_STEP + 1e-5,
                f"largest applied step over all phases {max_applied[0] * 1e3:.3f} mm / "
                f"{math.degrees(max_applied[1]):.3f} deg (caps {CAP_STEP * 1e3:.1f} mm / "
                f"{math.degrees(CAP_ROT_STEP):.2f} deg)",
            ),
        }
        slot_base_ok = all(
            error < (SLOT_BASE_ROT_TOL if name in ROT_ERRORS else SLOT_BASE_POS_TOL)
            for name, error in obs_errors.items()
        )
        results["slot-base observations = independent kinematics"] = (
            slot_base_ok and obs_checked > 0 and limits_checked > 0,
            f"largest differences over {obs_checked} policy steps (limits: only the {limits_checked} of "
            f"{obs_checked} steps without the clamp): "
            + ", ".join(
                f"{name} {error:.1e} rad" if name in ROT_ERRORS else f"{name} {error * 1e3:.5f} mm"
                for name, error in obs_errors.items()
            )
            + f" (tol {SLOT_BASE_POS_TOL * 1e3:.2f} mm, {SLOT_BASE_ROT_TOL:.0e} rad); largest observed setpoint offset "
            f"{max_setpoint_offset:.3f} of the bound",
        )
        rot_ok = (
            float(rot_follow.max()) <= ROT_FOLLOW_TOL
            and float(rot_overshoot.clamp(min=0).max()) <= ROT_OVERSHOOT_TOL
            and float(slot_base_drift.max()) <= ROT_DRIFT_TOL
            and float(rot_end.max()) <= ORIENTATION_TOL
            and rot_margin >= JOINT_MARGIN_MIN
        )
        results["follows the rotation"] = (
            rot_ok,
            f"about +z, {math.degrees(float(turned.min())):.1f} deg at up to {math.degrees(rot_speed):.2f} deg/step: "
            f"following {math.degrees(float(rot_follow.max())):.2f} deg, overshoot "
            f"{math.degrees(float(rot_overshoot.clamp(min=0).max())):.2f} deg, slot-base drift "
            f"{float(slot_base_drift.max()) * 1e3:.2f} mm, end {math.degrees(float(rot_end.max())):.2f} deg; "
            f"joint margin {rot_margin:.2f} rad",
        )
        print(f"\n=== check_push_env ({n} envs, policy at {1 / env.step_dt:.2f} Hz, damping {term.cfg.damping}) ===")
        env.close()
        check_report.finish(results)


if __name__ == "__main__":
    main()
