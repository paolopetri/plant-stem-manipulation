"""Sanity check of the FR3 + end-effector asset in simulation (PhysX, the current backend of `fr3_cfg`).

Loads `fr3_cfg(<ee>)` (default `fork_v2`), holds the start pose for a few hundred physics steps and checks:
- the articulation has the bodies fr3_link0..fr3_link7 + the end-effector;
- the end-effector stays rigidly at 107 mm along z of fr3_link7 (flange offset);
- the start pose is held (reports the per-joint deviation; known ~0.05 rad sag at joints 2 and 4 without
  gravity compensation, see the gravity question in docs/TODO.md).

Usage (from the repo root; runs headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/check_fr3.py
    uv run --extra isaacsim python scripts/check_fr3.py --ee fork
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Sanity check of the FR3 + end-effector asset.")
parser.add_argument("--ee", default="fork_v2", help="End-effector folder name in assets/fr3/end_effectors/.")
parser.add_argument("--steps", type=int, default=200, help="Physics steps to hold the start pose.")
parser.add_argument("--max_joint_dev", type=float, default=0.1, help="Allowed start-pose deviation [rad].")
add_launcher_args(parser)
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse

from stem_manip.assets.fr3 import fr3_cfg

FLANGE_OFFSET = torch.tensor([0.0, 0.0, 0.107])  # fr3_link7 -> fr3_link8 [m], FR3 kinematics
POS_TOL = 1e-3  # [m]


def main() -> None:
    """Spawn the robot, hold the start pose and print a pass/fail summary."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        cfg = fr3_cfg(args_cli.ee)
        cfg.prim_path = "/World/Robot"
        robot = Articulation(cfg)
        sim.reset()

        q0 = robot.data.default_joint_pos.torch.clone()
        for _ in range(args_cli.steps):
            robot.set_joint_position_target_index(target=q0)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())

        results = {}
        expected_bodies = [f"fr3_link{i}" for i in range(8)] + [args_cli.ee]
        results["bodies"] = (robot.body_names == expected_bodies, f"{robot.body_names}")

        pos = robot.data.body_pos_w.torch[0].cpu()
        quat = robot.data.body_quat_w.torch[0].cpu()
        i7, i_ee = robot.body_names.index("fr3_link7"), robot.body_names.index(args_cli.ee)
        ee_in_link7 = quat_apply_inverse(quat[i7:i7 + 1], (pos[i_ee] - pos[i7]).unsqueeze(0))[0]
        ee_err = float((ee_in_link7 - FLANGE_OFFSET).norm())
        results["ee rigid at flange"] = (ee_err < POS_TOL, f"{ee_in_link7.tolist()} (error {ee_err:.2e} m)")

        dev = (robot.data.joint_pos.torch - q0)[0].cpu()
        max_dev = float(dev.abs().max())
        results["start pose held"] = (
            max_dev < args_cli.max_joint_dev,
            f"max |q - q0| = {max_dev:.3f} rad, per joint {[round(float(d), 3) for d in dev]}",
        )

        print(f"\n=== check_fr3 ({type(sim_cfg.physics).__name__}, ee={args_cli.ee}) ===")
        for name, (ok, info) in results.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        print("=== all passed ===" if all(ok for ok, _ in results.values()) else "=== FAILED ===")


if __name__ == "__main__":
    main()
