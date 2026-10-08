"""Env check for stage 1 (`StemManip-Push-Position-FR3-v0`): the env steps, and the fork follows the actions.

Creates the env with a few envs, resets it and drives the tool tip with constant actions (same in every env):
hold (zero action), then full translation along +x, +y, +z, each for `--steps` policy steps and followed by a hold;
then a full rotation about the vertical (+z) for `--steps` policy steps and a hold. Then, each after a reset to the
start pose: a full +y translation and a full rotation about +z from rest (step limits), and a move +x followed by a
full move down (-z) into the ground and a lift (target clamp, restart after contact). The action term (6-D:
translation and rotation step, robot base frame) is Franka's Cartesian impedance law with an
integrated target, speed cap, step-change limit and target-offset clamp (`mdp.ToolTipImpedanceAction`). With the
step-change limit the target ramps up at the start of a move and brakes during the hold.
Checks:
- shapes: observation (num_envs, 21) and action (num_envs, 6), all observations finite;
- start pose: tool tip within 5 mm of (0.30, 0, 0.55) m and within 1 deg of its orientation at reset after the hold;
- holds still: the tool tip moves less than 1 mm during the last half of the hold;
- follows the actions: following error (largest distance to the target over the move and the hold; at the end of
  a policy step the interpolated target equals the step's end target) <= 2 mm; the tool passes the target's final
  position by at most 5 mm and ends within 1 mm of it; sideways drift < 2 mm; orientation within 1 deg throughout.
  The final target is the target after the clamp; that every commanded step was executed follows from the following
  error: had the clamp shortened a step, target and tool would be 4 mm apart at the end of that policy step;
- follows the rotation: about +z, angle following <= 2 deg, overshoot past the final target <= 2 deg, tool-tip
  drift while turning <= 2 mm, end within 1 deg of the target after the hold;
- step limits: from rest the first step is exactly 0.08 mm (+y translation) and 0.02 deg (rotation about +z), and
  the step then reaches the caps 3.2 mm and 1.44 deg;
- follows at full speed: over the +y ramp (incl. `FULL_SPEED_STEPS` steps at the cap, 10 cm/s) following error
  <= 2 mm; over the rotation ramp (up to 45 deg/s) angle following <= 2 deg and tool-tip drift <= 2 mm (the moves
  above stay below the caps because of the step-change limit);
- target clamp: at every physics step of the move down, the target the law uses is at most 4 mm from the tool; while
  the tool is blocked by the ground (last `BLOCKED_STEPS` policy steps) exactly 4 mm;
- restart after contact: lifting off the ground, the applied step changes by at most 0.08 mm per policy step;
- applied step within the caps (3.2 mm, 1.44 deg) in every phase.
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

import stem_manip.tasks  # noqa: F401  (registers the task)
from stem_manip.tasks.push_position.env_cfg import StemPushPositionEnvCfg
from stem_manip.utils.impedance import rotvec_between

TASK = "StemManip-Push-Position-FR3-v0"
START_POS = (0.30, 0.0, 0.55)  # [m] tool tip of the start pose (env_cfg.START_JOINT_POS)
START_TOL = 5e-3  # [m]
ORIENTATION_TOL = math.radians(1.0)
HOLD_TOL = 1e-3  # [m]
FOLLOW_TOL = 2e-3  # [m] following error while moving
OVERSHOOT_TOL = 5e-3  # [m]
END_TOL = 1e-3  # [m] after the hold following a move
DRIFT_TOL = 2e-3  # [m]
ROT_FOLLOW_TOL, ROT_OVERSHOOT_TOL, ROT_DRIFT_TOL = math.radians(2.0), math.radians(2.0), 2e-3  # user, 2026-10-07
OBS_DIM = 21  # tool-tip position 3, orientation 6, applied step 6, target offset 6
# decided action limits (option B1, user 2026-10-07/08; docs/overleaf_folder/open_questions/action_limits_problem.tex);
# fixed here, not read from the cfg, so that a wrong cfg value fails (with an override flag these checks fail too)
FIRST_STEP, FIRST_ROT_STEP = 8e-5, 0.000349  # [m], [rad] step-change limits 0.08 mm (2026-10-08) / 0.02 deg
CAP_STEP, CAP_ROT_STEP = 0.0032, 0.02513  # [m], [rad] speed caps 10 cm/s (2026-10-08) / 45 deg/s (2026-10-07)
CLAMP_OFFSET = 4e-3  # [m] target clamp, 4 N at K_p 1000 (2026-10-08)
JOINT_MARGIN_MIN = 0.25  # [rad] closer to a joint limit, a check is invalid (user, 2026-10-08)
# clamp test path (test setup, not a criterion): +x first, so that the way down stays clear of joint 4's limit
# (at 10 cm/s the tool reaches the ground after ~186 down steps; DOWN_STEPS leaves the blocked window well after it)
CLAMP_APPROACH_STEPS, DOWN_STEPS, BLOCKED_STEPS, LIFT_STEPS = 40, 260, 20, 10
FULL_SPEED_STEPS = 10  # policy steps at the translation cap at the end of the +y ramp


def main() -> None:
    """Run the phases and print a pass/fail summary."""
    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.episode_length_s = 1e4  # no time-out reset in the middle of the phases
    if args_cli.rot_stiffness is not None:
        env_cfg.actions.tool_tip.stiffness_rot = args_cli.rot_stiffness
    if args_cli.max_step is not None:
        env_cfg.actions.tool_tip.max_step = args_cli.max_step
    if args_cli.max_rot_step is not None:
        env_cfg.actions.tool_tip.max_rot_step = args_cli.max_rot_step
    if args_cli.damping is not None:
        env_cfg.actions.tool_tip.damping = args_cli.damping
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        obs, _ = env.reset()
        sim, device, n = env.sim, env.device, env.num_envs
        term = env.action_manager.get_term("tool_tip")
        quat_ref = term.tool_pose()[1].clone()  # start orientation, held while the rotation action is zero

        def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Angle between orientations a and b (n, 4) [rad] (axis-angle of a b^-1; exact also near 0)."""
            return rotvec_between(a, b).norm(dim=-1)

        def angle_to_fixed(quat: torch.Tensor) -> torch.Tensor:
            return angle_between(quat, quat_ref)

        step_time = env.step_dt * args_cli.slow_motion
        frame_start = time.perf_counter()
        finite = bool(torch.isfinite(obs["policy"]).all())
        max_angle = 0.0
        robot = env.scene[term.cfg.asset_name]
        joint_ids = robot.find_joints(term.cfg.joint_names)[0]
        limits = robot.data.joint_pos_limits.torch[:, joint_ids]  # (n, joints, 2)
        margin = math.inf  # smallest distance to a joint limit [rad] since the last `window()`
        max_applied = [0.0, 0.0]  # largest applied translation [m] and rotation [rad] step over all phases

        def window() -> None:
            """Start a measurement window of the joint-limit guard."""
            nonlocal margin
            margin = math.inf

        def step(action: torch.Tensor, track_tilt: bool = True) -> None:
            nonlocal frame_start, finite, max_angle, margin
            obs, _, _, _, _ = env.step(action.repeat(n, 1))
            finite = finite and bool(torch.isfinite(obs["policy"]).all())
            q = robot.data.joint_pos.torch[:, joint_ids]
            margin = min(margin, float(torch.minimum(q - limits[..., 0], limits[..., 1] - q).min()))
            applied = term.processed_actions
            max_applied[0] = max(max_applied[0], float(applied[:, :3].norm(dim=-1).max()))
            max_applied[1] = max(max_applied[1], float(applied[:, 3:].norm(dim=-1).max()))
            if track_tilt:
                max_angle = max(max_angle, float(angle_to_fixed(term.tool_pose()[1]).max()))
            if sim.is_rendering:
                time.sleep(max(0.0, step_time - (time.perf_counter() - frame_start)))
                frame_start = time.perf_counter()

        obs_shape, action_shape = tuple(obs["policy"].shape), tuple(env.action_manager.action.shape)
        pos0 = term.tool_pose()[0].clone()

        # -- hold: zero action
        zero = torch.zeros(6, device=device)
        for i in range(args_cli.steps):
            step(zero)
            if i == args_cli.steps // 2 - 1:
                pos_mid = term.tool_pose()[0].clone()
        pos_hold, quat_hold = (v.clone() for v in term.tool_pose())
        hold_move = float((pos_hold - pos_mid).norm(dim=-1).max())
        start_error = float((pos_hold - torch.tensor(START_POS, device=device)).norm(dim=-1).max())
        hold_angle = float(angle_to_fixed(quat_hold).max())

        # -- full action along +x, +y, +z, each followed by a hold
        moves = []
        window()
        for axis in range(3):
            action = torch.zeros(6, device=device)
            action[axis] = 1.0
            start = term.tool_pose()[0].clone()
            follow = torch.zeros(n, device=device)
            positions = []  # tool positions during the hold
            for i in range(args_cli.steps + args_cli.hold_steps):
                step(action if i < args_cli.steps else zero)
                follow = torch.maximum(follow, (term.tool_pose()[0] - term.target()).norm(dim=-1))
                if i >= args_cli.steps:
                    positions.append(term.tool_pose()[0][:, axis].clone())
            end = term.target().clone()  # final target (the target brakes during the hold)
            overshoot = (torch.stack(positions) - end[:, axis]).max(dim=0).values
            pos = term.tool_pose()[0]
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

        # -- rotation about the vertical (+z), reported only
        rotate = torch.zeros(6, device=device)
        rotate[5] = 1.0
        moves_margin = margin
        window()
        tip_start = term.tool_pose()[0].clone()
        rot_follow, tip_drift = torch.zeros(n, device=device), torch.zeros(n, device=device)
        rot_speed = 0.0  # largest applied rotation step [rad]
        quats = []  # tool orientations during the hold
        for i in range(args_cli.steps + args_cli.hold_steps):
            step(rotate if i < args_cli.steps else zero, track_tilt=False)
            tip_drift = torch.maximum(tip_drift, (term.tool_pose()[0] - tip_start).norm(dim=-1))
            rot_follow = torch.maximum(rot_follow, angle_between(term.tool_pose()[1], term.target_quat()))
            rot_speed = max(rot_speed, float(term.processed_actions[:, 3:].norm(dim=-1).max()))
            if i >= args_cli.steps:
                quats.append(term.tool_pose()[1].clone())
        target_end = term.target_quat().clone()
        # overshoot: rotation from the final target to the tool, about +z (the turning direction)
        rot_overshoot = torch.stack([rotvec_between(q, target_end)[:, 2] for q in quats]).max(0).values
        rot_end = angle_between(term.tool_pose()[1], target_end)
        turned = angle_between(target_end, quat_ref)

        rot_margin = margin

        def restart() -> None:
            """Reset to the start pose and hold (zero action) until the tool settles."""
            env.reset()
            for _ in range(args_cli.steps):
                step(zero, track_tilt=False)

        # -- step limits: from rest at the start pose, full +y translation, then full rotation about +z; the step
        # grows by the step-change limit until the cap
        def ramp(axis: int, n_steps: int) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, torch.Tensor]]:
            """Full action on `axis` from rest; returns the first and last applied step, the joint margin and the
            largest following error, angle following and tool-tip drift over the ramp."""
            restart()
            window()
            action = torch.zeros(6, device=device)
            action[axis] = 1.0
            taken = []
            tip_start = term.tool_pose()[0].clone()
            follow = {k: torch.zeros(n, device=device) for k in ("pos", "angle", "drift")}
            for _ in range(n_steps):
                step(action, track_tilt=False)
                taken.append(term.processed_actions.clone())
                pos, quat = term.tool_pose()
                follow["pos"] = torch.maximum(follow["pos"], (pos - term.target()).norm(dim=-1))
                follow["angle"] = torch.maximum(follow["angle"], angle_between(quat, term.target_quat()))
                follow["drift"] = torch.maximum(follow["drift"], (pos - tip_start).norm(dim=-1))
            return taken[0], taken[-1], margin, follow

        ramp_steps = math.ceil(CAP_STEP / FIRST_STEP) + FULL_SPEED_STEPS
        rot_ramp_steps = math.ceil(CAP_ROT_STEP / FIRST_ROT_STEP) + 2
        first, last, ramp_margin, ramp_follow = ramp(1, ramp_steps)
        first_rot, last_rot, rot_ramp_margin, rot_ramp_follow = ramp(5, rot_ramp_steps)

        # -- target clamp: +x, then full move down into the ground; at every physics step the target the law uses may
        # be at most CLAMP_OFFSET ahead of the tool (measured around each apply_actions call)
        restart()
        x_move = torch.zeros(6, device=device)
        x_move[0] = 1.0
        for i in range(CLAMP_APPROACH_STEPS + args_cli.hold_steps):
            step(x_move if i < CLAMP_APPROACH_STEPS else zero, track_tilt=False)
        stretch = torch.zeros(n, device=device)  # largest target-tool distance over the physics steps of a policy step
        apply_actions = term.apply_actions

        def apply_and_measure() -> None:
            pos = term.tool_pose()[0].clone()  # the pose the law sees in this physics step
            apply_actions()
            stretch.copy_(torch.maximum(stretch, (term.command_pose()[0] - pos).norm(dim=-1)))

        term.apply_actions = apply_and_measure
        down = torch.zeros(6, device=device)
        down[2] = -1.0
        stretches = []
        for i in range(DOWN_STEPS):
            if i == DOWN_STEPS - BLOCKED_STEPS:
                window()
            stretch.zero_()
            step(down, track_tilt=False)
            stretches.append(stretch.clone())
        term.apply_actions = apply_actions
        stretches = torch.stack(stretches)  # (DOWN_STEPS, n)
        blocked = stretches[-BLOCKED_STEPS:]
        clamp_margin = margin
        tip_height = term.tool_pose()[0][:, 2]
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
                start_error < START_TOL and hold_angle < ORIENTATION_TOL,
                f"tool tip {start_error * 1e3:.1f} mm from {START_POS}, orientation {math.degrees(hold_angle):.2f} deg "
                f"(at reset: {[round(float(v), 3) for v in pos0[0]]} m)",
            ),
            "holds still": (hold_move < HOLD_TOL, f"tool tip moved {hold_move * 1e3:.2f} mm in the 2nd half of the hold"),
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
                f"{math.degrees(float(rot_ramp_follow['angle'].max())):.2f} deg, tool-tip drift "
                f"{float(rot_ramp_follow['drift'].max()) * 1e3:.2f} mm",
            ),
            "target clamp": (
                bool((stretches <= CLAMP_OFFSET + 1e-6).all())
                and bool(((blocked - CLAMP_OFFSET).abs() < 1e-6).all())
                and clamp_margin >= JOINT_MARGIN_MIN,
                f"moving down into the ground: largest target-tool distance {float(stretches.max()) * 1e3:.4f} mm "
                f"(bound {CLAMP_OFFSET * 1e3:.1f} mm), while blocked {float(blocked.min()) * 1e3:.4f}-"
                f"{float(blocked.max()) * 1e3:.4f} mm, tool tip at z {float(tip_height.min()) * 1e3:.1f}-"
                f"{float(tip_height.max()) * 1e3:.1f} mm; joint margin {clamp_margin:.2f} rad",
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
        rot_ok = (
            float(rot_follow.max()) <= ROT_FOLLOW_TOL
            and float(rot_overshoot.clamp(min=0).max()) <= ROT_OVERSHOOT_TOL
            and float(tip_drift.max()) <= ROT_DRIFT_TOL
            and float(rot_end.max()) <= ORIENTATION_TOL
            and rot_margin >= JOINT_MARGIN_MIN
        )
        results["follows the rotation"] = (
            rot_ok,
            f"about +z, {math.degrees(float(turned.min())):.1f} deg at up to {math.degrees(rot_speed):.2f} deg/step: "
            f"following {math.degrees(float(rot_follow.max())):.2f} deg, overshoot "
            f"{math.degrees(float(rot_overshoot.clamp(min=0).max())):.2f} deg, tool-tip drift "
            f"{float(tip_drift.max()) * 1e3:.2f} mm, end {math.degrees(float(rot_end.max())):.2f} deg; joint margin "
            f"{rot_margin:.2f} rad",
        )
        print(f"\n=== check_push_env ({n} envs, policy at {1 / env.step_dt:.2f} Hz, damping {term.cfg.damping}) ===")
        for name, (ok, info) in results.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        print("=== all passed ===" if all(ok for ok, _ in results.values()) else "=== FAILED ===")
        env.close()


if __name__ == "__main__":
    main()
