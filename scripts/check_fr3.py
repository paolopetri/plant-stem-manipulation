"""Sanity check of the FR3 + end-effector asset in simulation (PhysX, the current backend of `fr3_cfg`).

Loads `fr3_cfg(<ee>, control="torque")` (default `fork_v2`; the task's mode: no joint PD, the joints take efforts),
holds the start pose with zero torque, then applies a torque pulse (+`--torque` on every joint, then -`--torque`)
and checks:
- the build is up to date: the slot_base pose in build/fr3_<ee>.urdf equals the end-effector yaml's (else "stale build,
  rerun build_asset");
- the articulation has the bodies fr3_link0..fr3_link7 + the end-effector;
- the end-effector stays rigidly at 107 mm along z of fr3_link7 (flange offset), at the start pose and after the pulse;
- robot gravity off (user, 2026-10-14): PhysX reports gravity disabled for every robot body, the end-effector
  included (the model of `fr3_cfg`: the real FR3 compensates gravity itself);
- torque mode (user, 2026-10-14): under a +-0.5 N m pulse every joint moves by more than 0.01 rad (with the position
  mode's joint PD, stiffness 400 N m/rad, the pulse would deflect a joint by 1.25 mrad);
- slot base = datasheet kinematics: the slot-base pose the task computes (`stem_manip.utils.slot_base.slot_base_pose`)
  equals the slot base from the measured joint angles via the FR3 Denavit-Hartenberg parameters and the yaml's
  slot_base (`stem_manip.utils.fr3_reference`), at the start pose and after the pulse.
  Tolerance 1e-5 m and 1e-5 rad (user, 2026-10-14): both sides use the same nominal values (datasheet = URDF), so
  only float32 rounding of the joint angles and link poses remains, about 1e-7 per transform, ~1e-6 over the chain;
  10x margin.

Usage (from the repo root; runs headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/check_fr3.py
    uv run --extra isaacsim python scripts/check_fr3.py --ee fork
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Sanity check of the FR3 + end-effector asset.")
parser.add_argument("--ee", default="fork_v2", help="End-effector folder name in assets/fr3/end_effectors/.")
parser.add_argument("--steps", type=int, default=200, help="Physics steps to hold the start pose.")
parser.add_argument("--torque", type=float, default=0.5, help="Joint torque of the pulse [N m].")
parser.add_argument("--pulse_steps", type=int, default=30, help="Physics steps per sign of the pulse.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse

from stem_manip.assets.fr3 import fr3_cfg, slot_base_offset, stale_build
from stem_manip.utils import check_report, slot_base
from stem_manip.utils.fr3_reference import rotvec_from_matrix, slot_base_pose

FLANGE_OFFSET = torch.tensor([0.0, 0.0, 0.107])  # fr3_link7 -> fr3_link8 [m], FR3 kinematics
POS_TOL = 1e-3  # [m] end-effector at the flange
MIN_PULSE_MOVE = 0.01  # [rad] per joint under the +-0.5 N m pulse (user, 2026-10-14)
SLOT_BASE_POS_TOL, SLOT_BASE_ROT_TOL = 1e-5, 1e-5  # [m], [rad] slot base vs datasheet kinematics (user, 2026-10-14)


def main() -> None:
    """Spawn the robot, hold the start pose, apply the pulse and print a pass/fail summary."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        cfg = fr3_cfg(args_cli.ee, control="torque")
        cfg.prim_path = "/World/Robot"
        robot = Articulation(cfg)
        sim.reset()
        # start at the start pose, as the task's reset event does (the spawned USD has all joints at 0)
        q0 = robot.data.default_joint_pos.torch.clone()
        robot.write_joint_position_to_sim_index(position=q0)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
        i7, i_ee = robot.body_names.index("fr3_link7"), robot.body_names.index(args_cli.ee)
        offset_pos, offset_quat = (torch.tensor([v], device=sim.device) for v in slot_base_offset(args_cli.ee))

        def run(torque: float, steps: int) -> None:
            for _ in range(steps):
                robot.set_joint_effort_target_index(target=torch.full_like(robot.data.joint_pos.torch, torque))
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.get_physics_dt())

        def ee_at_flange() -> float:
            """Distance of the end-effector body from 107 mm along z of fr3_link7 [m]."""
            pos, quat = robot.data.body_pos_w.torch[0].cpu(), robot.data.body_quat_w.torch[0].cpu()
            ee_in_link7 = quat_apply_inverse(quat[i7 : i7 + 1], (pos[i_ee] - pos[i7]).unsqueeze(0))[0]
            return float((ee_in_link7 - FLANGE_OFFSET).norm())

        def slot_base_error() -> tuple[float, float]:
            """Slot base as the task computes it vs the datasheet kinematics at the measured joints: position [m] and
            angle [rad] differences."""
            data = robot.data
            slot_base_pos, slot_base_quat = slot_base.slot_base_pose(  # the task's code (under test)
                data.root_pos_w.torch,
                data.root_quat_w.torch,
                data.body_pos_w.torch[:, i_ee],
                data.body_quat_w.torch[:, i_ee],
                offset_pos,
                offset_quat,
            )
            reference = slot_base_pose(data.joint_pos.torch, args_cli.ee)
            pos_err = (slot_base_pos.double() - reference[:, :3, 3]).norm(dim=-1)
            rot_err = rotvec_from_matrix(
                matrix_from_quat(slot_base_quat).double() @ reference[:, :3, :3].transpose(1, 2)
            )
            return float(pos_err.max()), float(rot_err.norm(dim=-1).max())

        gravity_off = robot.root_view.get_disable_gravities()
        gravity_off = gravity_off if isinstance(gravity_off, torch.Tensor) else torch.as_tensor(gravity_off.numpy())
        gravity_off = gravity_off.reshape(-1).bool().cpu().tolist()[: robot.num_bodies]
        run(0.0, args_cli.steps)
        ee_errors = [ee_at_flange()]
        slot_base_errors = [slot_base_error()]
        q_held = robot.data.joint_pos.torch.clone()
        run(args_cli.torque, args_cli.pulse_steps)
        run(-args_cli.torque, args_cli.pulse_steps)
        moved = (robot.data.joint_pos.torch - q_held)[0].cpu()
        ee_errors.append(ee_at_flange())
        slot_base_errors.append(slot_base_error())

        results = {}
        stale = stale_build(args_cli.ee)
        results["build up to date"] = (
            stale is None,
            "slot_base in the built URDF = yaml" if stale is None else f"stale build, rerun build_asset: {stale}",
        )
        expected_bodies = [f"fr3_link{i}" for i in range(8)] + [args_cli.ee]
        results["bodies"] = (robot.body_names == expected_bodies, f"{robot.body_names}")
        results["ee rigid at flange"] = (
            max(ee_errors) < POS_TOL,
            f"error at the start pose {ee_errors[0]:.2e} m, after the pulse {ee_errors[1]:.2e} m",
        )
        results["robot gravity off"] = (
            len(gravity_off) == robot.num_bodies and all(gravity_off),
            f"gravity disabled per body: {dict(zip(robot.body_names, gravity_off, strict=False))}",
        )
        results["torque mode"] = (
            bool((moved.abs() > MIN_PULSE_MOVE).all()),
            f"pulse +-{args_cli.torque} N m x {args_cli.pulse_steps} steps: joints moved "
            f"{[round(float(d), 3) for d in moved]} rad (each > {MIN_PULSE_MOVE} rad)",
        )
        results["slot base = datasheet kinematics"] = (
            all(p < SLOT_BASE_POS_TOL and r < SLOT_BASE_ROT_TOL for p, r in slot_base_errors),
            "; ".join(
                f"{where}: {p * 1e3:.5f} mm, {r:.1e} rad"
                for where, (p, r) in zip(("start pose", "after the pulse"), slot_base_errors, strict=True)
            )
            + f" (tol {SLOT_BASE_POS_TOL * 1e3:.2f} mm, {SLOT_BASE_ROT_TOL:.0e} rad)",
        )

        print(f"\n=== check_fr3 ({type(sim_cfg.physics).__name__}, ee={args_cli.ee}, torque mode) ===")
        check_report.finish(results)


if __name__ == "__main__":
    main()
