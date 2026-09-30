"""Sanity check of stem model v1 alone on Newton (no robot).

Spawns `stem_cfg()` in a few envs on a ground plane, and checks that the values of `assets/stem/stem.yaml`
arrive in the simulator:
- number of segments; start poses (segment centres on a vertical line above the base, tangent = local +Z up);
- total mass against rho * A * L;
- per-joint stretch / bend / twist stiffness in the Newton model against E A / l, E I / l (l = segment length).
Then steps the simulation and reports the tip height and the maximum curvature (must stay finite).

Not yet (docs/TODO.md -> M1): damping read-back; base fixed; stem stands and sags plausibly, springs back,
deflects when pushed; cantilever sag against delta = q L^4 / (8 E I); axial-strain noise.

Usage (from the repo root; headless unless a visualizer is requested, e.g. `--viz newton_gl`):
    uv run --extra isaacsim python scripts/check_stem.py
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Sanity check of the stem model.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments.")
parser.add_argument("--steps", type=int, default=200, help="Simulation steps to run.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import math

import torch
from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, CableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from stem_manip.assets.stem import stem_cfg, stem_params
from stem_manip.utils import stem_geometry

SIM_DT = 0.01  # [s]
NUM_SUBSTEPS = 8
VBD_ITERATIONS = 20
ENV_SPACING = 1.0  # [m]
POS_TOL = 1e-4  # [m]
REL_TOL = 1e-3


@configclass
class StemSceneCfg(InteractiveSceneCfg):
    """Ground plane and one stem per env."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    stem: CableObjectCfg = stem_cfg().replace(prim_path="{ENV_REGEX_NS}/Stem")


def _close(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= REL_TOL * abs(expected)


def main() -> None:
    """Spawn the stem, compare the simulator state with the yaml and print a pass/fail summary."""
    params = stem_params()
    geometry, material = params["geometry"], params["material"]
    num_segments = geometry["num_segments"]
    segment_length = geometry["length"] / num_segments
    area = math.pi * geometry["diameter"] ** 2 / 4
    area_moment = math.pi * geometry["diameter"] ** 4 / 64

    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        device=args_cli.device,
        physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=VBD_ITERATIONS), num_substeps=NUM_SUBSTEPS),
    )
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(1.5, 1.5, 0.8), target=(0.5, 0.0, 0.2))
        scene = InteractiveScene(StemSceneCfg(num_envs=args_cli.num_envs, env_spacing=ENV_SPACING))
        sim.reset()
        stem = scene["stem"]
        model = SimulationManager.get_model()
        segment_lengths = torch.full((num_segments,), segment_length, device=sim.device)
        results = {}

        # -- start state
        poses = stem.data.segment_pose_w.torch.clone()
        results["segments"] = (stem.num_segments == num_segments, f"{stem.num_segments} (yaml {num_segments})")

        base = scene.env_origins + torch.tensor(geometry["base_position"], device=sim.device)
        expected_pos = base.unsqueeze(1).repeat(1, num_segments, 1)
        expected_pos[..., 2] += (torch.arange(num_segments, device=sim.device) + 0.5) * segment_length
        pos_err = float((poses[..., :3] - expected_pos).norm(dim=-1).max())
        up = torch.zeros_like(poses[..., :3])
        up[..., 2] = 1.0
        tangent_err = float((quat_apply(poses[..., 3:], up) - up).norm(dim=-1).max())
        results["start poses"] = (
            pos_err < POS_TOL and tangent_err < 1e-4,
            f"max centre error {pos_err:.1e} m, max tangent error {tangent_err:.1e}",
        )

        # -- parameters in the Newton model (the stems are the only bodies and joints in the scene)
        mass = float(model.body_mass.numpy().sum()) / args_cli.num_envs
        expected_mass = material["density"] * area * geometry["length"]
        results["mass"] = (_close(mass, expected_mass), f"{mass * 1e3:.3f} g (rho A L = {expected_mass * 1e3:.3f} g)")

        twist_modulus = material["twist_modulus"]  # null -> Newton uses the bend stiffness for twist
        expected_ke = {
            "stretch E A / l": material["stretch_modulus"] * area / segment_length,
            "bend E I / l": material["bend_modulus"] * area_moment / segment_length,
            "twist G J / l": (
                material["bend_modulus"] * area_moment / segment_length
                if twist_modulus is None
                else twist_modulus * 2.0 * area_moment / segment_length
            ),
        }
        model_ke = sorted({float(v) for v in model.joint_target_ke.numpy()})
        ke_ok = all(any(_close(v, e) for v in model_ke) for e in expected_ke.values()) and all(
            any(_close(v, e) for e in expected_ke.values()) for v in model_ke
        )
        expected_info = ", ".join(f"{name} = {value:.4g}" for name, value in expected_ke.items())
        model_info = sorted({f"{v:.4g}" for v in model_ke}, key=float)
        results["joint stiffness"] = (ke_ok, f"model {model_info}; expected {expected_info}")

        # -- step
        for _ in range(args_cli.steps):
            sim.step(render=False)
            scene.update(SIM_DT)
            if sim.is_rendering:
                sim.render()
        poses = stem.data.segment_pose_w.torch
        curvature = stem_geometry.joint_curvature(poses, segment_lengths)
        tip_height = stem_geometry.point_pose(poses, num_segments - 1, 0.5 * segment_length)[:, 2]
        results[f"finite after {args_cli.steps} steps"] = (
            bool(torch.isfinite(poses).all()),
            f"tip height {[round(float(z), 4) for z in tip_height]} m, max curvature {float(curvature.max()):.3f} 1/m",
        )

        print(f"\n=== check_stem ({args_cli.num_envs} envs) ===")
        for name, (ok, info) in results.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        print("=== all passed ===" if all(ok for ok, _ in results.values()) else "=== FAILED ===")


if __name__ == "__main__":
    main()
