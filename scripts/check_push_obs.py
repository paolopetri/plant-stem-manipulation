"""Env check for stage 1 (`StemManip-Push-Position-FR3-v0`): the target command and the stem-state observations.

Creates the env with a few envs and holds the fork still at its start pose (zero action; the fork does not touch the
stem). Phases: rest (the stem settles), push (a constant sideways force on the tip segment, +x in the world frame),
release, then a reset. The observation is read by its decided layout (user, 2026-10-08):
tool tip 21 | stem base 3 | 5 stem points of the previous policy step 15 | the same 5 points now 15 | target 3,
all positions in the robot base frame [m]; the points at arc lengths 0.08, 0.16, 0.24, 0.32, 0.40 m (the last one is
the tip, the stem point the policy controls).
Checks:
- shapes: observation (num_envs, 57), all finite;
- stem at rest: the base point at the stem's base position (stem.yaml, robot base frame) within 1 mm; the 5 points on
  the vertical through it at their arc lengths within 1 mm;
- observed tip = tip from the segment poses (segment 19, +L/2), transformed into the robot base frame, within 1e-5 m,
  at rest and while pushed;
- previous step: the previous-step slice equals the current slice of the step before, at every step;
- pushed stem: with the force on the tip segment in +x every point moves in +x, higher points more, the tip by more
  than 1 cm; sideways (y) by less than 1 mm;
- target in the region: horizontal distance from the tip's rest position in [0.03, 0.10] m, height on or below the
  bowl (drop 0.6 r^2 / s, the stem pushed at its tip) and above the deepest shape within 0.8 x 5 1/m
  (`stem_target.deepest_drop_factor`); constant during the episode; different between envs; all different after a
  reset.
Visual check: the target (command marker) and the 5 observed points (small spheres, drawn by this script).

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/check_push_obs.py
    uv run --extra isaacsim python scripts/check_push_obs.py --num_envs 4 --viz kit --slow_motion 3
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Command and stem-observation check for the stage-1 push task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs.")
parser.add_argument("--rest_steps", type=int, default=30, help="Policy steps at rest before the measurement.")
parser.add_argument("--push_steps", type=int, default=60, help="Policy steps with the force on the tip segment.")
parser.add_argument("--release_steps", type=int, default=60, help="Policy steps after the force is removed.")
parser.add_argument("--push_force", type=float, default=0.3, help="Force on the tip segment, +x [N].")
parser.add_argument(
    "--slow_motion", type=float, default=1.0, help="With a viewer: play this many times slower than real time."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import time

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms

import stem_manip.tasks  # noqa: F401  (registers the task)
from stem_manip.assets.stem import stem_model, stem_params
from stem_manip.tasks.push_position.env_cfg import STEM_MODEL, StemPushPositionEnvCfg
from stem_manip.utils.stem_geometry import point_pose
from stem_manip.utils.stem_target import deepest_drop_factor

TASK = "StemManip-Push-Position-FR3-v0"
# decided layout and values (user, 2026-10-08); fixed here, not read from the cfg, so that a wrong cfg fails
OBS_DIM = 57
BASE, PREV, CURR, TARGET = slice(21, 24), slice(24, 39), slice(39, 54), slice(54, 57)
ARC_LENGTHS = (0.08, 0.16, 0.24, 0.32, 0.40)  # [m]
TIP_SEGMENT = 19
DISTANCE_RANGE = (0.03, 0.10)  # [m]
TARGET_MAX_CURVATURE = 0.8 * 5.0  # [1/m] budget 0.8 (2026-10-09) x damage.max_curvature
REST_TOL = 1e-3  # [m] points at rest vs the straight upright stem
FRAME_TOL = 1e-5  # [m] observed tip vs the tip from the segment poses
MIN_TIP_PUSH = 0.01  # [m]
SIDEWAYS_TOL = 1e-3  # [m]


def main() -> None:
    """Run the phases and print a pass/fail summary."""
    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.episode_length_s = 1e4  # no time-out reset in the middle of the phases
    env_cfg.commands.stem_target.resampling_time_range = (2e4, 2e4)  # above the episode length (sampled at reset)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        obs, _ = env.reset()
        sim, device, n = env.sim, env.device, env.num_envs
        robot, stem = env.scene["robot"], env.scene["stem"]
        model = stem_model(STEM_MODEL)
        geometry = stem_params(STEM_MODEL)["geometry"]
        segment_length = geometry["length"] / geometry["num_segments"]
        arc = torch.tensor(ARC_LENGTHS, device=device)

        # expected stem base in the robot base frame: stem.yaml's base position (env frame) minus the robot root
        root_env = robot.data.root_pos_w.torch - env.scene.env_origins
        base_expected = torch.tensor(geometry["base_position"], device=device) - root_env  # (n, 3)

        points_vis = None
        if sim.is_rendering:
            points_vis = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/StemPoints",
                    markers={
                        "point": sim_utils.SphereCfg(
                            radius=0.008, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0))
                        )
                    },
                )
            )

        def tip_from_segments() -> torch.Tensor:
            """Tip from the segment poses, robot base frame (n, 3): independent of the observation code."""
            tip_w = point_pose(model.segment_poses(stem), TIP_SEGMENT, 0.5 * segment_length)[:, :3]
            return subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, tip_w)[0]

        step_time = env.step_dt * args_cli.slow_motion
        frame_start = time.perf_counter()
        finite = bool(torch.isfinite(obs["policy"]).all())
        obs_shape = tuple(obs["policy"].shape)
        targets = [obs["policy"][:, TARGET].clone()]
        prev_error, frame_error = 0.0, 0.0
        zero = torch.zeros(n, 6, device=device)

        def step() -> torch.Tensor:
            """One policy step with zero action; checks the history and the frame; returns the policy obs."""
            nonlocal obs, frame_start, finite, prev_error, frame_error
            last = obs["policy"][:, CURR].clone()
            obs, _, _, _, _ = env.step(zero)
            policy = obs["policy"]
            finite = finite and bool(torch.isfinite(policy).all())
            prev_error = max(prev_error, float((policy[:, PREV] - last).abs().max()))
            frame_error = max(frame_error, float((policy[:, CURR][:, -3:] - tip_from_segments()).norm(dim=-1).max()))
            targets.append(policy[:, TARGET].clone())
            if points_vis is not None:
                root_w = robot.data.root_pos_w.torch  # robot base frame -> world (robot not rotated: checked below)
                points_vis.visualize((policy[:, CURR].reshape(n, 5, 3) + root_w.unsqueeze(1)).reshape(-1, 3))
                time.sleep(max(0.0, step_time - (time.perf_counter() - frame_start)))
                frame_start = time.perf_counter()
            return policy

        # -- rest
        for _ in range(args_cli.rest_steps):
            policy = step()
        base = policy[:, BASE]
        rest = policy[:, CURR].reshape(n, 5, 3)
        rest_expected = base_expected.unsqueeze(1).repeat(1, 5, 1)
        rest_expected[..., 2] += arc
        base_error = float((base - base_expected).norm(dim=-1).max())
        rest_error = float((rest - rest_expected).norm(dim=-1).max())
        robot_upright = bool((robot.data.root_quat_w.torch[:, 3].abs() > 1.0 - 1e-6).all())

        # -- push the tip segment in +x, then release
        set_force = model.segment_force_setter(stem)
        force = torch.zeros(n, 3, device=device)
        force[:, 0] = args_cli.push_force
        set_force(TIP_SEGMENT, force)
        for _ in range(args_cli.push_steps):
            policy = step()
        pushed = policy[:, CURR].reshape(n, 5, 3) - rest
        set_force(TIP_SEGMENT, None)
        for _ in range(args_cli.release_steps):
            step()

        # -- target: region, constant, different between envs and after a reset
        target = targets[0]
        tip_rest = base_expected.clone()
        tip_rest[:, 2] += ARC_LENGTHS[-1]
        distance = (target[:, :2] - tip_rest[:, :2]).norm(dim=-1)
        drop_factor = (tip_rest[:, 2] - target[:, 2]) * ARC_LENGTHS[-1] / distance**2  # 0.6 = the bowl
        deepest = deepest_drop_factor(distance, ARC_LENGTHS[-1], TARGET_MAX_CURVATURE)
        below_bowl = (drop_factor - 0.6) * distance**2 / ARC_LENGTHS[-1]  # [m]
        constant = float((torch.stack(targets) - target).abs().max())
        obs, _ = env.reset()
        after = obs["policy"][:, TARGET]
        distinct = n == 1 or torch.unique(target, dim=0).shape[0] == n

        results = {
            "shapes": (obs_shape == (n, OBS_DIM) and finite, f"observation {obs_shape}, all finite: {finite}"),
            "stem at rest": (
                base_error < REST_TOL and rest_error < REST_TOL and robot_upright,
                f"base point {base_error * 1e3:.3f} mm from {[round(float(v), 3) for v in base_expected[0]]} m; "
                f"points {rest_error * 1e3:.3f} mm from the upright stem (tol {REST_TOL * 1e3:.0f} mm); "
                f"robot root not rotated: {robot_upright}",
            ),
            "observed tip = tip from segment poses": (
                frame_error < FRAME_TOL,
                f"largest difference {frame_error * 1e3:.4f} mm (tol {FRAME_TOL * 1e3:.2f} mm)",
            ),
            "previous step": (prev_error < 1e-6, f"largest difference to the step before {prev_error:.2e} m"),
            "pushed stem": (
                bool((pushed[..., 0] > 0).all())
                and bool((pushed[..., 1:, 0] > pushed[..., :-1, 0]).all())
                and float(pushed[:, -1, 0].min()) > MIN_TIP_PUSH
                and float(pushed[..., 1].abs().max()) < SIDEWAYS_TOL,
                f"+x per point (env 0): {[round(float(v) * 1e3, 1) for v in pushed[0, :, 0]]} mm; "
                f"largest sideways {float(pushed[..., 1].abs().max()) * 1e3:.3f} mm",
            ),
            "target in the region": (
                bool((distance >= DISTANCE_RANGE[0] - 1e-6).all())
                and bool((distance <= DISTANCE_RANGE[1] + 1e-6).all())
                and bool((drop_factor >= 0.6 - 1e-3).all())
                and bool((drop_factor <= deepest + 1e-3).all())
                and constant == 0.0
                and distinct
                and bool(((after - target).norm(dim=-1) > 1e-6).all()),
                f"distance {float(distance.min()) * 1e3:.1f}-{float(distance.max()) * 1e3:.1f} mm, below the bowl "
                f"{float(below_bowl.min()) * 1e3:.2f}-{float(below_bowl.max()) * 1e3:.2f} mm (drop factor "
                f"{float(drop_factor.min()):.3f}-{float(drop_factor.max()):.3f}, deepest allowed "
                f"{[round(float(v), 3) for v in deepest]}); changed during the episode by {constant:.1e} m; distinct "
                f"between envs: {distinct}; resampled at reset: {bool(((after - target).norm(dim=-1) > 1e-6).all())}",
            ),
        }
        print(f"\n=== check_push_obs ({n} envs, push {args_cli.push_force} N on segment {TIP_SEGMENT}) ===")
        for name, (ok, info) in results.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        print("=== all passed ===" if all(ok for ok, _ in results.values()) else "=== FAILED ===")
        env.close()


if __name__ == "__main__":
    main()
