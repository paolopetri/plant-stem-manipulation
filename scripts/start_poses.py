"""Start poses of the push task (`env_cfg.START_POSES`, `env_cfg.EVAL_START_POSES`): solve them, or replay them.

Modes:
- `solve`: solves the start-pose table offline with differential IK in the simulator and prints it as Python, to be
  pasted into the env cfg. Goals: slot base beside the stem (stem axis at `START_POSE_STEM_XY`, upright), at
  `START_POSE_DISTANCE` from its axis, fork horizontal (slot-base z down), slot facing the stem axis; on a grid of
  slot-base heights in `START_POSE_HEIGHTS` (step `HEIGHT_STEP`) and azimuths within +-`START_POSE_AZIMUTH` (step
  `AZIMUTH_STEP`; azimuth about the stem axis, 0 = towards the robot, counterclockwise seen from above), plus the
  held-out goals `EVAL_GOALS`. Each goal is solved in its own env, seeded at the held-out pose `a_above_tip` (the
  closest solution: elbow up). The robot's joints are written directly at every iteration (kinematic, no dynamics).
  The IK is damped least squares with a null-space pull towards the seed (the arm's self-motion stays near it).
  A table goal is rejected when IK does not reach it (`POS_TOL`, `ANGLE_TOL`), when a joint is closer than
  `START_POSE_MIN_MARGIN` to its limit, or when an arm body (any robot body but the fork: surface points of the FR3
  collision meshes) is closer than `START_POSE_ARM_CLEARANCE` to the stem's surface. Joint angles rounded to 1 mrad.
- `pose`: one start pose, printed as the Hydra override line of the reset event's `poses` (to play a policy from it:
  README, "Push task: training"). `--height H --azimuth DEG` solves it like a table goal (same IK, goal and checks; the
  checks are printed, not enforced); `--held_out NAME` prints a committed held-out pose (`EVAL_START_POSES`) without
  the simulator.
- `replay` (viewer check): shows every table pose (and the held-out poses) one after the other for `--hold` seconds,
  with the stem at `START_POSE_STEM_XY`, and prints each pose's slot-base height, azimuth, joint margin and arm
  clearance.

Checked in `tests/test_fr3_reference.py` (independent FK on every committed pose).

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/start_poses.py solve
    uv run --extra isaacsim python scripts/start_poses.py pose --height 0.25 --azimuth 30
    uv run python scripts/start_poses.py pose --held_out b_side_90
    uv run --extra isaacsim python scripts/start_poses.py replay --viz kit
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Solve or replay the push task's start poses.")
parser.add_argument(
    "mode", choices=["solve", "pose", "replay"], help="Solve the table, one pose (override line), or replay the poses."
)
parser.add_argument("--height", type=float, default=None, help="pose: slot-base height [m].")
parser.add_argument("--azimuth", type=float, default=None, help="pose: azimuth about the stem axis [deg].")
parser.add_argument("--held_out", default=None, help="pose: name of a committed held-out pose, instead of solving.")
parser.add_argument("--hold", type=float, default=2.0, help="replay: seconds per pose.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import math
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_from_matrix

from stem_manip.assets.fr3 import ARM_LINKS, arm_surface_points, fr3_cfg, slot_base_offset
from stem_manip.assets.stem import stem_model, stem_params
from stem_manip.tasks.push_position import env_cfg as task
from stem_manip.tasks.push_position.mdp.slot_base import SlotBaseKinematics
from stem_manip.utils import stem_geometry
from stem_manip.utils.impedance import rotvec_between

HEIGHT_STEP = 0.05  # [m] table grid
AZIMUTH_STEP = math.radians(10.0)  # [rad] table grid
# held-out goals (slot-base height [m], azimuth [rad]); a_above_tip is the seed itself, not solved
EVAL_GOALS = {"b_side_90": (0.30, math.radians(90.0)), "c_low": (0.08, math.radians(60.0))}
POS_TOL, ANGLE_TOL = 2e-3, math.radians(0.5)  # [m], [rad] IK reached its goal (after rounding to 1 mrad)
MAX_ITERS = 800
MAX_JOINT_STEP = 0.02  # [rad] per IK iteration
DAMPING = 0.01  # damped least squares
NULLSPACE_GAIN = 0.1  # per iteration, towards the seed
MESH_POINTS = 4000  # surface points per FR3 collision mesh for the arm clearance


def goal_pose(height: float, azimuth: float) -> tuple[tuple[float, float, float], torch.Tensor]:
    """Slot-base goal beside the reference stem: position (robot base frame) and rotation matrix (3, 3); fork
    horizontal, slot facing the stem axis."""
    direction = (math.cos(azimuth), math.sin(azimuth))  # slot base -> stem axis, horizontal
    stem_x, stem_y = task.START_POSE_STEM_XY
    position = (
        stem_x - task.START_POSE_DISTANCE * direction[0],
        stem_y - task.START_POSE_DISTANCE * direction[1],
        height,
    )
    fork_x = (direction[0], direction[1], 0.0)
    fork_y = (direction[1], -direction[0], 0.0)  # z x x with z = down
    return position, torch.tensor((fork_x, fork_y, (0.0, 0.0, -1.0))).T


def override_line(q: list[float]) -> str:
    """Hydra override of the reset event's `poses` with the single pose q [rad], quoted for the shell."""
    return "'env.events.start_pose.params.poses=[[" + ",".join(f"{v:.3f}" for v in q) + "]]'"


def main() -> None:
    if args_cli.mode == "pose" and args_cli.held_out is not None:
        print(override_line(task.EVAL_START_POSES[args_cli.held_out]))
        return
    if args_cli.mode == "pose" and (args_cli.height is None or args_cli.azimuth is None):
        parser.error("pose needs --height and --azimuth, or --held_out")
    stem_module = stem_model(task.STEM_MODEL)
    geometry = stem_params(task.STEM_MODEL)["geometry"]
    stem_radius, stem_length = geometry["diameter"] / 2, geometry["length"]
    if args_cli.mode == "pose":
        goals = {"pose": (args_cli.height, math.radians(args_cli.azimuth))}
        names = list(goals)
    elif args_cli.mode == "solve":
        heights = torch.arange(
            task.START_POSE_HEIGHTS[0], task.START_POSE_HEIGHTS[1] + 1e-6, HEIGHT_STEP, dtype=torch.float64
        ).tolist()
        num_azimuths = round(task.START_POSE_AZIMUTH / AZIMUTH_STEP)
        azimuths = [k * AZIMUTH_STEP for k in range(-num_azimuths, num_azimuths + 1)]
        goals = {f"h{h:.2f}_az{math.degrees(a):+.0f}": (h, a) for h in heights for a in azimuths}
        goals.update(EVAL_GOALS)
        names = list(goals)
    else:
        poses = {f"table_{i:02d}": q for i, q in enumerate(task.START_POSES)} | task.EVAL_START_POSES
        names = list(poses)

    sim_cfg = sim_utils.SimulationCfg(dt=0.002, device=args_cli.device, physics=stem_module.physics_cfg())
    with launch_simulation(sim_cfg, args_cli):
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils import configclass

        num_envs = len(names) if args_cli.mode != "replay" else 1

        @configclass
        class StartPoseSceneCfg(InteractiveSceneCfg):
            """Ground, light, FR3 with the fork (joint PD), and in `replay` the stem at the reference position."""

            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            light = AssetBaseCfg(
                prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
            )
            robot = fr3_cfg(task.END_EFFECTOR).replace(prim_path="{ENV_REGEX_NS}/Robot")

        scene_cfg = StartPoseSceneCfg(num_envs=num_envs, env_spacing=2.0)
        if args_cli.mode == "replay":
            scene_cfg.stem = stem_module.stem_cfg().replace(prim_path="{ENV_REGEX_NS}/Stem")
            scene_cfg.stem.init_state.pos = (*task.START_POSE_STEM_XY, 0.0)
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(1.3, -1.1, 0.8), target=(0.45, 0.0, 0.25))
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        robot, device = scene["robot"], sim.device
        arm = robot.find_joints(task.ARM_JOINT_NAMES, preserve_order=True)[0]
        kinematics = SlotBaseKinematics(robot, task.END_EFFECTOR, slot_base_offset(task.END_EFFECTOR), arm)
        limits = robot.data.joint_pos_limits.torch[:, arm]
        link_ids = [robot.body_names.index(name) for name in ARM_LINKS]
        # surface points of each link's collision mesh, in the link frame
        link_points = [p.to(device) for p in arm_surface_points(MESH_POINTS)]
        origins = scene.env_origins

        def write(q: torch.Tensor) -> None:
            """Put the arm at q (num_envs, 7) and let the simulator update the link poses (one step, held by the
            joint PD at q)."""
            robot.write_joint_position_to_sim_index(position=q, joint_ids=arm)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=arm)
            robot.set_joint_position_target_index(target=q, joint_ids=arm)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())

        def margin() -> torch.Tensor:
            """Smallest distance of the arm joints to a limit (num_envs,) [rad]."""
            q = robot.data.joint_pos.torch[:, arm]
            return torch.minimum(q - limits[..., 0], limits[..., 1] - q).amin(dim=-1)

        def arm_clearance() -> tuple[torch.Tensor, list[str]]:
            """Smallest distance of the arm's collision surfaces to the reference stem's surface (num_envs,) [m], and
            the closest link per env."""
            pos = robot.data.body_pos_w.torch - origins.unsqueeze(1)  # robot base at the env origin, unrotated
            quat = robot.data.body_quat_w.torch
            stem_axis = torch.tensor(
                [[[*task.START_POSE_STEM_XY, stem_length / 2, 0.0, 0.0, 0.0, 1.0]]], device=device
            ).expand(num_envs, -1, -1)  # one upright segment: the straight stem
            per_link = []
            for b, p in zip(link_ids, link_points, strict=True):
                points = quat_apply(quat[:, b : b + 1].expand(-1, len(p), -1), p.expand(num_envs, -1, -1))
                distance = stem_geometry.distance_to_centre_line(stem_axis, points + pos[:, b : b + 1], stem_length)
                per_link.append(distance.amin(dim=-1))
            clearance, closest = torch.stack(per_link, dim=-1).min(dim=-1)
            return clearance - stem_radius, [ARM_LINKS[i] for i in closest.tolist()]

        def azimuth_height() -> tuple[torch.Tensor, torch.Tensor]:
            pos = kinematics.pose()[0]
            rel = pos[:, :2] - torch.tensor(task.START_POSE_STEM_XY, device=device)
            return torch.atan2(-rel[:, 1], -rel[:, 0]), pos[:, 2]

        seed = torch.tensor(task.EVAL_START_POSES["a_above_tip"], device=device).repeat(num_envs, 1)
        write(seed)

        if args_cli.mode != "replay":
            targets = [goal_pose(*goals[name]) for name in names]
            goal_pos = torch.tensor([t[0] for t in targets], device=device)
            goal_quat = quat_from_matrix(torch.stack([t[1] for t in targets]).to(device))
            eye = torch.eye(7, device=device)
            for _ in range(MAX_ITERS):
                pos, quat = kinematics.pose()
                q = robot.data.joint_pos.torch[:, arm]
                error = torch.cat([goal_pos - pos, rotvec_between(goal_quat, quat)], dim=-1)
                jacobian = kinematics.jacobian()
                jacobian_t = jacobian.transpose(1, 2)
                damped = jacobian @ jacobian_t + DAMPING**2 * torch.eye(6, device=device)
                pinv = jacobian_t @ torch.linalg.inv(damped)  # damped least squares
                step = (pinv @ error.unsqueeze(-1)).squeeze(-1)
                pull = NULLSPACE_GAIN * (seed - q)  # self-motion towards the seed: the closest solution
                step = step + ((eye - pinv @ jacobian) @ pull.unsqueeze(-1)).squeeze(-1)
                write(q + step.clamp(-MAX_JOINT_STEP, MAX_JOINT_STEP))
            q = torch.round(robot.data.joint_pos.torch[:, arm] * 1000.0) / 1000.0
            write(q)  # the rounded angles are the ones committed: judge those
            pos, quat = kinematics.pose()
            pos_err, angle_err = (pos - goal_pos).norm(dim=-1), rotvec_between(quat, goal_quat).norm(dim=-1)
            margins, (clearances, closest) = margin(), arm_clearance()
            if args_cli.mode == "pose":
                print(
                    f"height {args_cli.height:.3f} m, azimuth {args_cli.azimuth:+.1f} deg: pos error "
                    f"{float(pos_err[0]) * 1e3:.2f} mm, angle error {math.degrees(angle_err[0]):.2f} deg, margin "
                    f"{float(margins[0]):.2f} rad (table: >= {task.START_POSE_MIN_MARGIN}), arm clearance "
                    f"{float(clearances[0]) * 100:.1f} cm ({closest[0]}; table: >= "
                    f"{task.START_POSE_ARM_CLEARANCE * 100:.0f} cm)"
                )
                print(override_line(q[0].tolist()))
                return
            table, held_out = [], {}
            for i, name in enumerate(names):
                height, azimuth = goals[name]
                reached = bool(pos_err[i] < POS_TOL and angle_err[i] < ANGLE_TOL)
                ok = reached and margins[i] >= task.START_POSE_MIN_MARGIN
                ok = ok and clearances[i] >= task.START_POSE_ARM_CLEARANCE
                print(
                    f"{name:16s} pos {float(pos_err[i]) * 1e3:6.2f} mm, angle {math.degrees(angle_err[i]):5.2f} deg, "
                    f"margin {float(margins[i]):.2f} rad, arm clearance {float(clearances[i]) * 100:5.1f} cm "
                    f"({closest[i]})"
                    f"  {'ok' if ok else 'rejected'}"
                )
                values = ", ".join(f"{v:.3f}" for v in q[i].tolist())
                comment = (
                    f"slot base {height:.2f} m, {math.degrees(azimuth):+.0f} deg; margin {float(margins[i]):.2f} rad"
                )
                if name in EVAL_GOALS:
                    held_out[name] = (f"({values}),", comment + ("" if reached else " NOT REACHED"))
                elif ok:
                    table.append((f"({values}),", comment))
            print(f"\n# {len(table)} of {len(names) - len(EVAL_GOALS)} table goals accepted")
            print("START_POSES = (")
            for values, comment in table:
                print(f"    {values}  # {comment}")
            print(")")
            for name, (values, comment) in held_out.items():
                print(f'    "{name}": {values}  # {comment}')
            return

        # -- replay: every pose for --hold seconds
        frame = 1.0 / 30.0
        for name in names:
            q = torch.tensor([poses[name]], device=device)
            write(q)
            azimuth, height = azimuth_height()
            clearance, closest = arm_clearance()
            print(
                f"{name:12s} slot base {float(height[0]):.3f} m, {math.degrees(azimuth[0]):+6.1f} deg; margin "
                f"{float(margin()[0]):.2f} rad, arm clearance {float(clearance[0]) * 100:.1f} cm "
                f"({closest[0]})"
            )
            for _ in range(max(1, round(args_cli.hold / frame))):
                write(q)
                if sim.is_rendering:
                    sim.render()
                    time.sleep(frame)


if __name__ == "__main__":
    main()
