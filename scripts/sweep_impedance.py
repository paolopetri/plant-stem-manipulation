"""Sweep: how well the fork follows step-wise commands under Franka's Cartesian impedance law (sim study).

Question: with a task-space impedance controller (the one the real FR3 will run), how fast can the fork move and
still follow the commanded path, and what does that do to the stem? Background and results:
docs/overleaf_folder/open_questions/impedance_action_study.tex.

Controller (`stem_manip.utils.impedance`, the same functions as the env's action term): the law of Franka's example
controller (franka_ros `cartesian_impedance_example_controller`, same task
part as libfranka's `cartesian_impedance_control.cpp`), evaluated every simulation step:
    tau = J^T [-K_p e_p - D_p (v - v_d);  -K_o e_o - D_o w]  +  (I - J^T J^T+) (k_ns (q_ns - q) - 2 sqrt(k_ns) dq)
with D = 2 sqrt(K) (Franka's choice), e_o from the quaternion error as in Franka's code, the torque change limited
to 1 N m per ms (Franka's `delta_tau_max`), and no Coriolis term (not available from PhysX here; at < 0.2 m/s it
is small). Joint armature: the motors' reflected inertia from the FR3 URDF (`--armature`; the robot asset has
none, and without it the simulated joints are far lighter than the real ones). Robot gravity is off in simulation (the real robot compensates it). v_d = 0 except for `plai_ff`.

Variants (how one policy action, a tool-tip displacement per policy step at 31.25 Hz, becomes the target):
- `rel_current`: target = measured tool tip + action, held over the policy step (Isaac Lab's relative OSC
  action, FrankaTwin);
- `plai`: target = previous target + action, held (IndustReal's policy-level action integrator);
- `plai_ff`: as `plai`, but the target moves linearly through the policy step and its velocity is fed forward
  (v_d = action / policy step).
Each env runs one (variant, stiffness K_p, speed) combination; all envs run in one simulation.

Protocol (same geometry as `check_contact.py --test slot`: fork horizontal at 0.30 m, slot towards the stem,
yaw 45 deg): approach with the joint PD (as in check_contact), switch the arm to torque control, then the
scripted "policy" commands (1) 10 cm straight back, away from the stem, at the env's speed, hold 1 s; (2) 35 cm
forward: 10 cm back to the start, then 17 cm prong reach + 3 cm gap until the slot bottom touches the stem, and 5 cm
push; hold 1.5 s.

Reported per env (`RESULT` lines and a table):
- `lag_ss`: mean distance between the tool tip and the commanded path over the last quarter of the free move
  (steady following);
- `lag`: largest distance between the tool tip and the commanded path at the same time, during the free move;
- `overshoot`: how far the tool tip passes the end of the free move;
- `err_free`: tool tip to commanded end point after the 1 s hold (free space);
- `err_push`: the same at the end of the push hold (the stem pushes back on the fork);
- `F_max`: largest contact force fork -> stem [N]; `kappa_max`: largest stem curvature [1/m] (limit in stem.yaml).

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/sweep_impedance.py
    uv run --extra isaacsim python scripts/sweep_impedance.py --variants plai_ff --gains 400 --speeds 0.1 \
        --viz kit --slow_motion 3
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

VARIANTS = ("rel_current", "plai", "plai_ff")

parser = argparse.ArgumentParser(description="Impedance-controller sweep: following step-wise commands.")
parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS), help="Target variants.")
parser.add_argument("--gains", nargs="+", type=float, default=[150.0, 400.0, 1000.0], help="K_p [N/m].")
parser.add_argument("--speeds", nargs="+", type=float, default=[0.05, 0.10, 0.20], help="Commanded speed [m/s].")
parser.add_argument("--rot_stiffness", type=float, default=10.0, help="K_o [N m/rad] (Franka default 10).")
parser.add_argument("--nullspace_stiffness", type=float, default=0.5, help="k_ns [N m/rad] (franka_ros 0.5).")
parser.add_argument(
    "--damping",
    choices=["franka", "apparent_mass"],
    default="franka",
    help="Task-space damping: Franka's 2 sqrt(K) or critical damping for the apparent mass (utils.impedance).",
)
parser.add_argument(
    "--armature",
    choices=["urdf", "none"],
    default="urdf",
    help="Joint armature in torque mode: reflected motor inertia from the FR3 URDF (gear ratio^2 x motor inertia) "
    "or none (the robot asset's value, 0).",
)
parser.add_argument(
    "--slow_motion", type=float, default=1.0, help="With a viewer: play this many times slower than real time."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import itertools
import math
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_from_matrix,
    skew_symmetric_matrix,
)

from stem_manip.assets.fr3 import fr3_cfg, tool_tip_offset
from stem_manip.assets.stem import stem_model, stem_params
from stem_manip.utils import stem_geometry
from stem_manip.utils.impedance import apparent_mass_damping, cartesian_impedance_torque, limit_torque_rate

PUSH_HEIGHT = 0.30  # [m] tool-tip height
FORK_YAW = math.radians(45.0)  # fork direction in the ground plane (as check_contact)
CLEAR_HEIGHT = 0.55  # [m] approach over the stem's tip
BACKOFF = 0.10  # [m] the approach goes down this far behind the start
APPROACH_SPEED = 0.10  # [m/s]
PRONG_REACH = 0.17  # [m] fork collision mesh beyond the tool tip along tool x
APPROACH_GAP = 0.03  # [m] gap between fork and stem at the start
FREE_MOVE = 0.10  # [m] straight back, away from the stem
PUSH_DEPTH = 0.05  # [m] fork travel beyond the stem surface
MOVE_TIME, SETTLE_TIME = 3.0, 1.0  # [s] approach to the point over the stem; settle before torque control
HOLD_FREE, HOLD_PUSH = 1.0, 1.5  # [s]
DECIMATION = 16  # sim steps per policy step (31.25 Hz with 2 ms)
# reflected motor inertia per arm joint [kg m^2] = gear_ratio^2 x motor_inertia, `<dynamics>` tags of the FR3 URDF
# (franka_description): joints 1-2: 120^2 x 4.206e-5, joints 3-4: 120^2 x 3.212e-5, joints 5-7: 80^2 x 3.212e-5
URDF_ARMATURE = (0.6057, 0.6057, 0.4625, 0.4625, 0.2055, 0.2055, 0.2055)
TORQUE_RATE = 1.0e3  # [N m/s] Franka's delta_tau_max = 1 N m per 1 ms
CONTACT_THRESHOLD = 1e-3  # [N]
RENDER_DT = 1.0 / 30.0  # [s]


def main() -> None:
    """Run all combinations in parallel envs and print one RESULT line per env plus a table."""
    stem_module = stem_model("chain")
    params = stem_params("chain")
    geometry = params["geometry"]
    radius = geometry["diameter"] / 2
    num_segments = geometry["num_segments"]
    segment_length = geometry["length"] / num_segments
    stem_x, stem_y = geometry["base_position"][:2]
    sim_dt = params["solver"]["sim_dt"]
    combos = list(itertools.product(args_cli.variants, args_cli.gains, args_cli.speeds))
    num_envs = len(combos)

    push_dir = (math.cos(FORK_YAW), math.sin(FORK_YAW), 0.0)
    start_offset = radius + PRONG_REACH + APPROACH_GAP  # stem axis to tool tip at the start, along the push

    def on_path(distance: float) -> tuple[float, float, float]:
        return (stem_x + distance * push_dir[0], stem_y + distance * push_dir[1], PUSH_HEIGHT)

    start = on_path(-start_offset)
    down = on_path(-start_offset - BACKOFF)
    over = (down[0], down[1], CLEAR_HEIGHT)

    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, physics=stem_module.physics_cfg())
    with launch_simulation(sim_cfg, args_cli):
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils import configclass

        @configclass
        class SweepSceneCfg(InteractiveSceneCfg):
            """Ground, light, FR3 with fork_v2, chain stem, contact sensor fork -> stem."""

            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            light = AssetBaseCfg(
                prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
            )
            robot = fr3_cfg("fork_v2").replace(prim_path="{ENV_REGEX_NS}/Robot")
            stem = stem_module.stem_cfg().replace(prim_path="{ENV_REGEX_NS}/Stem")
            fork_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Stem/seg_.*", filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*fork_v2"]
            )

        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(1.2, -0.9, 0.7), target=(0.45, 0.0, 0.25))
        scene = InteractiveScene(SweepSceneCfg(num_envs=num_envs, env_spacing=2.0))
        sim.reset()
        robot, stem, fork_sensor = scene["robot"], scene["stem"], scene["fork_contact"]
        device, origins = sim.device, scene.env_origins
        n = num_envs

        def per_env(values) -> torch.Tensor:
            return torch.tensor(values, device=device, dtype=torch.float32)

        variant = [c[0] for c in combos]
        k_pos, speed = per_env([c[1] for c in combos]), per_env([c[2] for c in combos])
        k_rot, k_ns = args_cli.rot_stiffness, args_cli.nullspace_stiffness
        is_rel = torch.tensor([v == "rel_current" for v in variant], device=device)
        is_ff = torch.tensor([v == "plai_ff" for v in variant], device=device)

        start_t, down_t, over_t, u = (per_env(v).repeat(n, 1) for v in (start, down, over, push_dir))
        fork_dir, side_dir = push_dir, (math.sin(FORK_YAW), -math.cos(FORK_YAW), 0.0)
        tool_quat = quat_from_matrix(per_env([fork_dir, side_dir, (0.0, 0.0, -1.0)]).T.unsqueeze(0)).repeat(n, 1)

        default_joint_pos = robot.data.default_joint_pos.torch.clone()
        robot.write_joint_position_to_sim_index(position=default_joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default_joint_pos))
        robot.set_joint_position_target_index(target=default_joint_pos)

        ee = robot.body_names.index("fork_v2")
        jacobian_body = ee - 1 if robot.is_fixed_base else ee
        arm = list(range(7))
        jacobian_columns = [robot.num_base_dofs + j for j in arm]
        offset_pos, offset_quat = (per_env(v).repeat(n, 1) for v in tool_tip_offset("fork_v2"))
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), n, device
        )

        def tool_pose() -> tuple[torch.Tensor, torch.Tensor]:
            """Tool-tip position (env frame) and orientation (robot root = env frame axes)."""
            pos, quat = robot.data.body_pos_w.torch[:, ee] - origins, robot.data.body_quat_w.torch[:, ee]
            return combine_frame_transforms(pos, quat, offset_pos, offset_quat)

        def tool_jacobian() -> torch.Tensor:
            """Geometric Jacobian (n, 6, 7) of the tool tip: linear rows shifted from the fork body."""
            jacobian = robot.data.body_link_jacobian_w.torch[:, jacobian_body, :, jacobian_columns].clone()
            lever = quat_apply(robot.data.body_quat_w.torch[:, ee], offset_pos)
            jacobian[:, 0:3, :] += torch.bmm(-skew_symmetric_matrix(lever), jacobian[:, 3:, :])
            return jacobian

        render_every = max(1, round(RENDER_DT / sim_dt))
        clock = {"steps": 0, "frame_start": time.perf_counter()}

        def advance() -> None:
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)
            clock["steps"] += 1
            if sim.is_rendering and clock["steps"] % render_every == 0:
                sim.render()
                frame_time = args_cli.slow_motion * render_every * sim_dt
                time.sleep(max(0.0, frame_time - (time.perf_counter() - clock["frame_start"])))
                clock["frame_start"] = time.perf_counter()

        # -- approach with the joint PD and differential IK (as check_contact), then settle
        def ik_step(target_pos: torch.Tensor, target_quat: torch.Tensor) -> None:
            pos, quat = tool_pose()
            ik.set_command(torch.cat([target_pos, target_quat], dim=-1))
            joint_pos = ik.compute(pos, quat, tool_jacobian(), robot.data.joint_pos.torch[:, arm])
            robot.set_joint_position_target_index(target=joint_pos, joint_ids=arm)
            advance()

        pos0, quat0 = (v.clone() for v in tool_pose())
        sign = torch.where((quat0 * tool_quat).sum(-1, keepdim=True) >= 0, 1.0, -1.0)
        move_steps = round(MOVE_TIME / sim_dt)
        for i in range(move_steps):
            s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / move_steps)
            quat = quat0 + s * (sign * tool_quat - quat0)
            ik_step(pos0 + s * (over_t - pos0), quat / quat.norm(dim=-1, keepdim=True))
        for leg_start, leg_end in ((over_t, down_t), (down_t, start_t)):
            leg_steps = round(float((leg_end - leg_start)[0].norm()) / APPROACH_SPEED / sim_dt)
            for i in range(leg_steps):
                s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / leg_steps)
                ik_step(leg_start + s * (leg_end - leg_start), tool_quat)
        for _ in range(round(SETTLE_TIME / sim_dt)):
            ik_step(start_t, tool_quat)

        # -- switch the arm to torque control (Franka's impedance law below)
        robot.write_joint_stiffness_to_sim_index(stiffness=0.0, joint_ids=arm)
        robot.write_joint_damping_to_sim_index(damping=0.0, joint_ids=arm)
        if args_cli.armature == "urdf":
            robot.write_joint_armature_to_sim_index(armature=per_env(URDF_ARMATURE).repeat(n, 1), joint_ids=arm)
        q_ns = robot.data.joint_pos.torch[:, arm].clone()  # null-space posture: the start configuration
        tau_prev = torch.zeros(n, 7, device=device)

        def impedance_step(x_d: torch.Tensor, v_d: torch.Tensor) -> None:
            nonlocal tau_prev
            pos, quat = tool_pose()
            jacobian = tool_jacobian()
            q, dq = robot.data.joint_pos.torch[:, arm], robot.data.joint_vel.torch[:, arm]
            twist = torch.bmm(jacobian, dq.unsqueeze(-1)).squeeze(-1)
            tau = torch.empty(n, 7, device=device)
            for gain in k_pos.unique():  # the law takes one stiffness; envs grouped by K_p
                ids = (k_pos == gain).nonzero().squeeze(-1)
                damping = None
                if args_cli.damping == "apparent_mass":  # PhysX's mass matrix excludes the armature: add it
                    mass = robot.data.mass_matrix.torch[ids][:, arm][:, :, arm]
                    mass = mass + torch.diag_embed(robot.data.joint_armature.torch[ids][:, arm])
                    damping = apparent_mass_damping(jacobian[ids], mass, float(gain), k_rot)
                tau[ids] = cartesian_impedance_torque(
                    jacobian[ids], pos[ids], quat[ids], twist[ids], q[ids], dq[ids], x_d[ids], v_d[ids],
                    tool_quat[ids], float(gain), k_rot, k_ns, q_ns[ids], damping,
                )
            tau = limit_torque_rate(tau, tau_prev, TORQUE_RATE * sim_dt)
            tau_prev = tau
            robot.set_joint_effort_target_index(target=tau, joint_ids=arm)
            advance()

        # commanded path c(t) per env: back FREE_MOVE, hold, forward FREE_MOVE + gap + depth, hold
        forward = FREE_MOVE + PRONG_REACH + APPROACH_GAP + PUSH_DEPTH  # tool tip (slot bottom) 5 cm past contact
        t_back, t_fwd = FREE_MOVE / speed, forward / speed
        segments = [(t_back, -FREE_MOVE), (HOLD_FREE, 0.0), (t_fwd, forward), (HOLD_PUSH, 0.0)]
        t_ends = torch.cumsum(torch.stack([torch.as_tensor(d, device=device).expand(n) for d, _ in segments]), 0)

        def commanded(t: float) -> tuple[torch.Tensor, torch.Tensor]:
            """Commanded path position (n, 3) and phase index (n,) at time t."""
            s = torch.zeros(n, device=device)
            t_start = torch.zeros(n, device=device)
            for (duration, length), t_end in zip(segments, t_ends):
                duration = torch.as_tensor(duration, device=device).expand(n)
                frac = ((t - t_start) / duration).clamp(0.0, 1.0)
                s = s + frac * length
                t_start = t_end
            phase = (t >= t_ends).sum(0)
            return start_t + s.unsqueeze(-1) * u, phase

        policy_dt = DECIMATION * sim_dt
        total_steps = math.ceil(float(t_ends[-1].max()) / sim_dt)
        x_d = start_t.clone()
        target_prev = start_t.clone()  # target at the start of the current policy step (plai, plai_ff)
        action = torch.zeros(n, 3, device=device)
        lag = torch.zeros(n, device=device)
        lag_sum, lag_count = torch.zeros(n, device=device), torch.zeros(n, device=device)
        overshoot = torch.zeros(n, device=device)
        free_end = start_t - FREE_MOVE * u
        err_free = torch.zeros(n, device=device)
        f_max = torch.zeros(n, device=device)
        kappa_max = torch.zeros(n, device=device)
        segment_lengths = torch.full((num_segments,), segment_length, device=device)
        for step_i in range(total_steps):
            t = step_i * sim_dt
            if step_i % DECIMATION == 0:  # policy step: the displacement it wants over the next policy step
                pos_now = tool_pose()[0]
                action = commanded(t + policy_dt)[0] - commanded(t)[0]
                target_prev = torch.where(is_rel.unsqueeze(-1), pos_now, x_d)
                x_d = target_prev + action
            frac = (step_i % DECIMATION + 1) / DECIMATION
            x_d_now = torch.where(is_ff.unsqueeze(-1), target_prev + frac * action, x_d)
            v_d = torch.where(is_ff.unsqueeze(-1), action / policy_dt, torch.zeros_like(action))
            impedance_step(x_d_now, v_d)

            pos = tool_pose()[0]
            path, phase = commanded(t + sim_dt)
            in_free = phase == 0
            lag = torch.where(in_free, torch.maximum(lag, (pos - path).norm(dim=-1)), lag)
            late = in_free & (t + sim_dt > 0.75 * t_back)  # last quarter of the free move: steady following
            lag_sum += torch.where(late, (pos - path).norm(dim=-1), 0.0)
            lag_count += late.float()
            in_hold_free = phase == 1
            passed = ((pos - free_end) * (-u)).sum(-1)  # > 0: beyond the end of the free move
            overshoot = torch.where(in_hold_free, torch.maximum(overshoot, passed), overshoot)
            err_free = torch.where(in_hold_free, (pos - free_end).norm(dim=-1), err_free)
            force = fork_sensor.data.force_matrix_w.torch.sum(dim=(1, 2)).norm(dim=-1)
            f_max = torch.maximum(f_max, force)
            poses = stem_module.segment_poses(stem)
            kappa_max = torch.maximum(kappa_max, stem_geometry.joint_curvature(poses, segment_lengths).amax(-1))
        lag_steady = lag_sum / lag_count.clamp(min=1)
        err_push = (tool_pose()[0] - commanded(float(t_ends[-1].max()))[0]).norm(dim=-1)

        print(f"\n=== sweep_impedance ({n} envs; damping {args_cli.damping}, K_o {k_rot} N m/rad, k_ns {k_ns}; "
              f"policy {1 / policy_dt:.2f} Hz) ===")
        header = f"{'variant':<12} {'K_p':>6} {'v':>5} {'lag_ss':>7} {'lag':>7} {'oversh.':>7} {'err_free':>8} {'err_push':>8} " \
                 f"{'F_max':>6} {'kappa':>6}"
        print(header)
        print(f"{'':<12} {'N/m':>6} {'m/s':>5} {'mm':>7} {'mm':>7} {'mm':>7} {'mm':>8} {'mm':>8} {'N':>6} {'1/m':>6}")
        for i, (var, gain, vel) in enumerate(combos):
            values = (lag_steady[i] * 1e3, lag[i] * 1e3, overshoot[i].clamp(min=0) * 1e3, err_free[i] * 1e3, err_push[i] * 1e3)
            print(f"{var:<12} {gain:6.0f} {vel:5.2f} " + " ".join(f"{float(v):7.2f}" for v in values[:3]) + " "
                  + " ".join(f"{float(v):8.2f}" for v in values[3:]) + f" {float(f_max[i]):6.2f} {float(kappa_max[i]):6.2f}")
        for i, (var, gain, vel) in enumerate(combos):
            print(f"RESULT variant={var} K_p={gain:.0f} speed={vel:.2f} lag_steady_mm={float(lag_steady[i]) * 1e3:.2f} lag_mm={float(lag[i]) * 1e3:.2f} "
                  f"overshoot_mm={float(overshoot[i].clamp(min=0)) * 1e3:.2f} err_free_mm={float(err_free[i]) * 1e3:.2f} "
                  f"err_push_mm={float(err_push[i]) * 1e3:.2f} F_max_N={float(f_max[i]):.3f} "
                  f"kappa_max={float(kappa_max[i]):.2f}")
        print(f"[INFO] curvature limit {params['damage']['max_curvature']} 1/m")


if __name__ == "__main__":
    main()
