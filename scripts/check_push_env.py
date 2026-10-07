"""Env check for stage 1 (`StemManip-Push-Position-FR3-v0`): the env steps, and the fork follows the actions.

Creates the env with a few envs, resets it and drives the tool tip with constant actions (same in every env):
hold (zero action), then full action along +x, +y, +z, each for `--steps` policy steps, each followed by a short
hold; finally a diagonal action (1, 1, 1) to check the step limit. The action term is Franka's Cartesian impedance
law with an integrated target (`mdp.ToolTipImpedanceAction`).
Checks:
- shapes: observation (num_envs, obs_dim) and action (num_envs, 3) as declared, all observations finite;
- start pose: tool tip within 5 mm of (0.30, 0, 0.55) m and within 1 deg of the fixed orientation after the hold;
- holds still: the tool tip moves less than 1 mm during the last half of the hold;
- follows the actions: steady following error (largest distance to the commanded target over the last quarter of
  the move; at the end of a policy step the interpolated target equals the step's end target) <= 2 mm; after each
  move the tool passes the end point by at most 5 mm and ends within 1 mm of it (every step executed); sideways
  drift < 2 mm; orientation within 1 deg throughout. The start-up peak (from rest to full speed in one step) is
  reported, not checked (same criterion as `scripts/sweep_impedance.py`, lag_ss);
- step limit: with action (1, 1, 1) the target moves by exactly max_step per policy step (length, not per axis).
Reported: per axis, travel vs. commanded, largest following error and overshoot.

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

TASK = "StemManip-Push-Position-FR3-v0"
START_POS = (0.30, 0.0, 0.55)  # [m] tool tip of the start pose (env_cfg.START_JOINT_POS)
START_TOL = 5e-3  # [m]
ORIENTATION_TOL = math.radians(1.0)
HOLD_TOL = 1e-3  # [m]
FOLLOW_TOL = 2e-3  # [m] following error while moving
OVERSHOOT_TOL = 5e-3  # [m]
END_TOL = 1e-3  # [m] after the hold following a move
DRIFT_TOL = 2e-3  # [m]


def main() -> None:
    """Run the phases and print a pass/fail summary."""
    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    if args_cli.rot_stiffness is not None:
        env_cfg.actions.tool_tip.stiffness_rot = args_cli.rot_stiffness
    if args_cli.max_step is not None:
        env_cfg.actions.tool_tip.max_step = args_cli.max_step
    if args_cli.damping is not None:
        env_cfg.actions.tool_tip.damping = args_cli.damping
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        obs, _ = env.reset()
        sim, device, n = env.sim, env.device, env.num_envs
        term = env.action_manager.get_term("tool_tip")
        max_step = term.cfg.max_step
        tool_quat = torch.tensor(term.cfg.tool_quat, device=device)

        def angle_to_fixed(quat: torch.Tensor) -> torch.Tensor:
            """Angle between each env's tool orientation and the fixed one [rad]."""
            return 2 * torch.acos((quat @ tool_quat).abs().clamp(max=1.0))

        step_time = env.step_dt * args_cli.slow_motion
        frame_start = time.perf_counter()
        finite = bool(torch.isfinite(obs["policy"]).all())
        max_angle = 0.0

        def step(action: torch.Tensor) -> None:
            nonlocal frame_start, finite, max_angle
            obs, _, _, _, _ = env.step(action.repeat(n, 1))
            finite = finite and bool(torch.isfinite(obs["policy"]).all())
            max_angle = max(max_angle, float(angle_to_fixed(term.tool_pose()[1]).max()))
            if sim.is_rendering:
                time.sleep(max(0.0, step_time - (time.perf_counter() - frame_start)))
                frame_start = time.perf_counter()

        obs_shape, action_shape = tuple(obs["policy"].shape), tuple(env.action_manager.action.shape)
        pos0 = term.tool_pose()[0].clone()

        # -- hold: zero action
        zero = torch.zeros(3, device=device)
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
        for axis in range(3):
            action = torch.zeros(3, device=device)
            action[axis] = 1.0
            start = term.tool_pose()[0].clone()
            follow, peak = torch.zeros(n, device=device), torch.zeros(n, device=device)
            for i in range(args_cli.steps):
                step(action)
                error = (term.tool_pose()[0] - term.target()).norm(dim=-1)
                peak = torch.maximum(peak, error)
                if i >= 0.75 * args_cli.steps:
                    follow = torch.maximum(follow, error)
            end = start.clone()
            end[:, axis] += args_cli.steps * max_step
            overshoot = torch.zeros(n, device=device)
            for _ in range(args_cli.hold_steps):
                step(zero)
                overshoot = torch.maximum(overshoot, term.tool_pose()[0][:, axis] - end[:, axis])
            pos = term.tool_pose()[0]
            delta = pos - start
            drift = torch.cat([delta[:, :axis], delta[:, axis + 1 :]], dim=1).norm(dim=-1)
            moves.append(
                {
                    "travel": delta[:, axis],
                    "follow": follow,
                    "peak": peak,
                    "overshoot": overshoot.clamp(min=0.0),
                    "end_error": (pos - end).norm(dim=-1),
                    "drift": drift,
                }
            )

        # -- diagonal action: the step's length is limited to max_step
        target_before = term.target().clone()
        step(torch.ones(3, device=device))
        diagonal_step = (term.target() - target_before).norm(dim=-1)

        follows = all(
            bool((m["follow"] <= FOLLOW_TOL).all())
            and bool((m["overshoot"] <= OVERSHOOT_TOL).all())
            and bool((m["end_error"] <= END_TOL).all())
            and bool((m["drift"] <= DRIFT_TOL).all())
            for m in moves
        )
        commanded = args_cli.steps * max_step
        results = {
            "shapes": (
                obs_shape[0] == n and action_shape == (n, 3) and finite,
                f"observation {obs_shape}, action {action_shape}, all finite: {finite}",
            ),
            "start pose": (
                start_error < START_TOL and hold_angle < ORIENTATION_TOL,
                f"tool tip {start_error * 1e3:.1f} mm from {START_POS}, orientation {math.degrees(hold_angle):.2f} deg "
                f"(at reset: {[round(float(v), 3) for v in pos0[0]]} m)",
            ),
            "holds still": (hold_move < HOLD_TOL, f"tool tip moved {hold_move * 1e3:.2f} mm in the 2nd half of the hold"),
            "follows the actions": (
                follows and max_angle < ORIENTATION_TOL,
                "; ".join(
                    f"{'xyz'[a]}: travel {float(m['travel'].min()) * 1e3:.1f} mm, following "
                    f"{float(m['follow'].max()) * 1e3:.2f} mm (start peak {float(m['peak'].max()) * 1e3:.2f}), overshoot {float(m['overshoot'].max()) * 1e3:.2f} mm, "
                    f"end {float(m['end_error'].max()) * 1e3:.2f} mm, drift {float(m['drift'].max()) * 1e3:.2f} mm"
                    for a, m in enumerate(moves)
                )
                + f"; commanded {commanded * 1e3:.1f} mm; max orientation error {math.degrees(max_angle):.2f} deg",
            ),
            "step limit": (
                bool(((diagonal_step - max_step).abs() < 1e-6).all()),
                f"action (1, 1, 1): target step {float(diagonal_step.max()) * 1e3:.3f} mm, max_step {max_step * 1e3:.3f} mm",
            ),
        }
        print(f"\n=== check_push_env ({n} envs, policy at {1 / env.step_dt:.2f} Hz, damping {term.cfg.damping}) ===")
        for name, (ok, info) in results.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        print("=== all passed ===" if all(ok for ok, _ in results.values()) else "=== FAILED ===")
        env.close()


if __name__ == "__main__":
    main()
