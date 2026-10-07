"""Sweep: does the push task's action follow its steps everywhere in the workspace (free space, no stem)?

Uses the env of `StemManip-Push-Position-FR3-v0` itself (its action term: Franka's Cartesian impedance law with an
integrated target), with the stem moved out of reach behind the robot. Each env first drives the tool tip to its own
start point (the 8 corners of the box below, plus the env's start pose), holds, and then moves `--steps` full steps
along +x, -x, +y, -y, +z, -z (each followed by a hold), and then turns by `ROT_MOVE` with full rotation steps about
+x, -x, +y, -y, +z, -z of the robot base frame (each followed by a hold).
Measured per start point and direction, against the criterion of
docs/overleaf_folder/open_questions/impedance_action_study.tex:
- following error: largest distance to the commanded target over the last quarter of the move (<= 2 mm);
- overshoot past the end point during the hold (<= 5 mm); end error after the hold (<= 1 mm);
- the fork's tilt from its orientation at the start of the move (<= 1 deg).
Rotation moves: following error (angle to the target orientation over the last quarter, <= 2 deg), overshoot past
the commanded angle (<= 2 deg), tool-tip drift while turning (the position target does not move, <= 2 mm); the end
error is reported.

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/sweep_action_poses.py
    uv run --extra isaacsim python scripts/sweep_action_poses.py --max_step 0.0025 --damping franka
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Action following over the workspace (free space).")
parser.add_argument("--steps", type=int, default=30, help="Policy steps per move.")
parser.add_argument("--hold_steps", type=int, default=45, help="Policy steps of zero action after each move.")
parser.add_argument("--max_step", type=float, default=None, help="Override the action's max_step [m].")
parser.add_argument("--max_rot_step", type=float, default=None, help="Override the action's max_rot_step [rad].")
parser.add_argument("--damping", choices=["franka", "apparent_mass"], default=None, help="Override the damping.")
parser.add_argument("--rot_stiffness", type=float, default=None, help="Override K_o [N m/rad].")
parser.add_argument("--nullspace_stiffness", type=float, default=None, help="Override k_ns [N m/rad].")
add_launcher_args(parser)
args_cli = parser.parse_args()

import itertools
import math

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_box_minus

import stem_manip.tasks  # noqa: F401  (registers the task)
from stem_manip.tasks.push_position.env_cfg import StemPushPositionEnvCfg

TASK = "StemManip-Push-Position-FR3-v0"
BOX_X, BOX_Y, BOX_Z = (0.35, 0.60), (-0.20, 0.20), (0.20, 0.45)  # [m] tool-tip start points, robot base frame
START_POSE = (0.30, 0.0, 0.55)  # [m] the env's start pose
STEM_AWAY = (-0.8, 0.8, 0.0)  # [m] stem base, out of the fork's reach behind the robot
ROT_MOVE = math.radians(20.0)  # [rad] each rotation move
FOLLOW_TOL, OVERSHOOT_TOL, END_TOL, TILT_TOL = 2e-3, 5e-3, 1e-3, math.radians(1.0)
ROT_FOLLOW_TOL, ROT_OVERSHOOT_TOL, ROT_DRIFT_TOL = math.radians(2.0), math.radians(2.0), 2e-3  # user, 2026-10-07
DIRECTIONS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def main() -> None:
    """Run all start points in parallel envs and print a table plus one RESULT line per start point."""
    starts = [*itertools.product(BOX_X, BOX_Y, BOX_Z), START_POSE]
    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = len(starts)
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.scene.stem.init_state.pos = STEM_AWAY
    env_cfg.episode_length_s = 1e4  # no time-out during the sweep
    action_cfg = env_cfg.actions.tool_tip
    for name, value in (("max_step", args_cli.max_step), ("max_rot_step", args_cli.max_rot_step), ("damping", args_cli.damping),
                        ("stiffness_rot", args_cli.rot_stiffness),
                        ("stiffness_nullspace", args_cli.nullspace_stiffness)):
        if value is not None:
            setattr(action_cfg, name, value)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        device, n = env.device, env.num_envs
        term = env.action_manager.get_term("tool_tip")
        max_step, max_rot_step = term.cfg.max_step, term.cfg.max_rot_step
        start_t = torch.tensor(starts, device=device)

        def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Angle between orientations a and b (n, 4) [rad]."""
            return quat_box_minus(a, b).norm(dim=-1)

        # -- drive every env's target to its start point, then hold
        while True:
            remaining = start_t - term.target()
            if float(remaining.norm(dim=-1).max()) < 1e-6:
                break
            env.step(torch.cat(((remaining / max_step).clamp(-1.0, 1.0), torch.zeros(n, 3, device=device)), dim=-1))
        for _ in range(2 * args_cli.hold_steps):
            env.step(torch.zeros(n, 6, device=device))
        settle = (term.tool_pose()[0] - start_t).norm(dim=-1)
        robot = env.scene["robot"]
        q, limits = robot.data.joint_pos.torch[:, :7], robot.data.joint_pos_limits.torch[:, :7]
        margin = torch.minimum(q - limits[..., 0], limits[..., 1] - q)  # (n, 7) [rad]
        margin_min, margin_joint = margin.min(dim=-1)

        results = {}  # direction -> dict of (n,) tensors
        for name, direction in DIRECTIONS.items():
            d = torch.tensor(direction, device=device, dtype=torch.float32)
            begin, begin_quat = term.target().clone(), term.target_quat().clone()
            follow, peak_tilt = torch.zeros(n, device=device), torch.zeros(n, device=device)
            for i in range(args_cli.steps):
                env.step(torch.cat((d, torch.zeros(3, device=device))).repeat(n, 1))
                peak_tilt = torch.maximum(peak_tilt, angle_between(term.tool_pose()[1], begin_quat))
                if i >= 0.75 * args_cli.steps:
                    follow = torch.maximum(follow, (term.tool_pose()[0] - term.target()).norm(dim=-1))
            end = begin + args_cli.steps * max_step * d
            overshoot = torch.zeros(n, device=device)
            for _ in range(args_cli.hold_steps):
                env.step(torch.zeros(n, 6, device=device))
                overshoot = torch.maximum(overshoot, ((term.tool_pose()[0] - end) * d).sum(-1))
            results[name] = {
                "follow": follow,
                "overshoot": overshoot.clamp(min=0.0),
                "end": (term.tool_pose()[0] - end).norm(dim=-1),
                "tilt": peak_tilt,
            }

        rot_steps = round(ROT_MOVE / max_rot_step)
        rot_results = {}
        for name, direction in DIRECTIONS.items():
            r = torch.tensor(direction, device=device, dtype=torch.float32)
            begin_pos, begin_quat = term.tool_pose()[0].clone(), term.target_quat().clone()
            follow, drift = torch.zeros(n, device=device), torch.zeros(n, device=device)
            for i in range(rot_steps):
                env.step(torch.cat((torch.zeros(3, device=device), r)).repeat(n, 1))
                drift = torch.maximum(drift, (term.tool_pose()[0] - begin_pos).norm(dim=-1))
                if i >= 0.75 * rot_steps:
                    follow = torch.maximum(follow, angle_between(term.tool_pose()[1], term.target_quat()))
            end_quat, commanded = term.target_quat().clone(), rot_steps * max_rot_step
            overshoot = torch.zeros(n, device=device)
            for _ in range(args_cli.hold_steps):
                env.step(torch.zeros(n, 6, device=device))
                overshoot = torch.maximum(overshoot, angle_between(term.tool_pose()[1], begin_quat) - commanded)
                drift = torch.maximum(drift, (term.tool_pose()[0] - begin_pos).norm(dim=-1))
            rot_results[name] = {
                "follow": follow,
                "overshoot": overshoot.clamp(min=0.0),
                "end": angle_between(term.tool_pose()[1], end_quat),
                "drift": drift,
            }

        print(f"\n=== sweep_action_poses ({n} start points; damping {term.cfg.damping}, K_p {term.cfg.stiffness_pos}, "
              f"K_o {term.cfg.stiffness_rot}, max_step {max_step * 1e3:.2f} mm = {max_step / env.step_dt * 100:.2f} cm/s, "
              f"max_rot_step {math.degrees(max_rot_step):.2f} deg = {math.degrees(max_rot_step) / env.step_dt:.0f} deg/s) ===")
        print(f"{'start point [m]':<22} {'settle':>7} {'follow':>14} {'overshoot':>14} {'end':>7} {'tilt':>12}")
        print(f"{'':<22} {'mm':>7} {'mm (worst dir)':>14} {'mm (worst dir)':>14} {'mm':>7} {'deg (dir)':>12}")
        all_ok = True
        for i, start in enumerate(starts):
            def worst(key: str) -> tuple[float, str]:
                name = max(results, key=lambda k: float(results[k][key][i]))
                return float(results[name][key][i]), name

            follow, follow_dir = worst("follow")
            overshoot, overshoot_dir = worst("overshoot")
            end, _ = worst("end")
            peak_tilt, tilt_dir = worst("tilt")
            ok = follow <= FOLLOW_TOL and overshoot <= OVERSHOOT_TOL and end <= END_TOL and peak_tilt <= TILT_TOL
            all_ok = all_ok and ok
            label = "(" + ", ".join(f"{v:.2f}" for v in start) + ")"
            print(f"{label:<22} {float(settle[i]) * 1e3:7.2f} {follow * 1e3:8.2f} ({follow_dir}) "
                  f"{overshoot * 1e3:8.2f} ({overshoot_dir}) {end * 1e3:7.2f} {math.degrees(peak_tilt):7.2f} ({tilt_dir})"
                  f"  limit margin {float(margin_min[i]):.2f} rad (joint {int(margin_joint[i]) + 1})  {'PASS' if ok else 'FAIL'}")
            print(f"RESULT start={label} follow_mm={follow * 1e3:.2f} overshoot_mm={overshoot * 1e3:.2f} "
                  f"end_mm={end * 1e3:.2f} tilt_deg={math.degrees(peak_tilt):.2f} ok={ok}")
        print(f"--- rotations of {math.degrees(rot_steps * max_rot_step):.1f} deg about the base axes ---")
        print(f"{'start point [m]':<22} {'follow':>14} {'overshoot':>14} {'end':>7} {'tip drift':>14}")
        print(f"{'':<22} {'deg (dir)':>14} {'deg (dir)':>14} {'deg':>7} {'mm (dir)':>14}")
        for i, start in enumerate(starts):
            def worst_rot(key: str) -> tuple[float, str]:
                name = max(rot_results, key=lambda k: float(rot_results[k][key][i]))
                return float(rot_results[name][key][i]), name

            follow, follow_dir = worst_rot("follow")
            overshoot, overshoot_dir = worst_rot("overshoot")
            end, _ = worst_rot("end")
            drift, drift_dir = worst_rot("drift")
            label = "(" + ", ".join(f"{v:.2f}" for v in start) + ")"
            ok = follow <= ROT_FOLLOW_TOL and overshoot <= ROT_OVERSHOOT_TOL and drift <= ROT_DRIFT_TOL
            all_ok = all_ok and ok
            print(f"{label:<22} {math.degrees(follow):8.2f} (r{follow_dir}) {math.degrees(overshoot):8.2f} (r{overshoot_dir}) "
                  f"{math.degrees(end):7.2f} {drift * 1e3:8.2f} (r{drift_dir})  {'PASS' if ok else 'FAIL'}")
            print(f"RESULT_ROT start={label} follow_deg={math.degrees(follow):.2f} overshoot_deg={math.degrees(overshoot):.2f} "
                  f"end_deg={math.degrees(end):.2f} drift_mm={drift * 1e3:.2f}")
        print("=== all passed ===" if all_ok else "=== FAILED ===")
        env.close()


if __name__ == "__main__":
    main()
