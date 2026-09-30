"""Sanity check of stem model v1 alone on Newton (no robot).

Spawns `stem_cfg()` in a few envs on a ground plane, and checks that the values of `assets/stem/stem.yaml`
arrive in the simulator:
- number of segments; start poses (segment centres on a vertical line above the base, tangent = local +Z up);
- total mass against rho * A * L;
- per-joint stretch / shear / bend / twist stiffness in the Newton model against E A / l, E I / l
  (l = segment length);
- per-joint damping in the Newton model against damping_time * stiffness (bend and twist only).
Then clamps the base (`fix_stem_base`), gives the upright stem a sideways velocity and steps the simulation:
- the base segment keeps its start pose;
- the stem deflects (tip moves sideways) and all poses stay finite;
- the stem springs back (tip returns to its start position);
- swing frequency: within 15 % of the first bending frequency of a clamped beam of the free length
  (the segment chain with its soft shear spring is a little softer than the ideal beam);
- damping ratio: within 0.02 of damping_time * pi * frequency (the solver adds about 0.01-0.02 of its own).
A second stem per env is clamped horizontally and sags under its own weight:
- cantilever sag: tip drop at the end within 5 % of the value computed by hand for the segment chain
  (bending + shear, `stem_reference.chain_tip_sag`).
Solver settings come from `solver` in the yaml.

Not yet (docs/TODO.md -> M1): deflects when pushed; axial-strain noise.

Usage (from the repo root; headless unless a visualizer is requested, e.g. `--viz newton_gl`):
    uv run --extra isaacsim python scripts/check_stem.py
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Sanity check of the stem model.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments.")
parser.add_argument("--steps", type=int, default=400, help="Simulation steps to run after the kick.")
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

from stem_manip.assets.stem import fix_stem_base, stem_cfg, stem_params
from stem_manip.utils import stem_geometry, stem_reference

ENV_SPACING = 1.0  # [m]
POS_TOL = 1e-4  # [m]
REL_TOL = 1e-3
KICK_TIP_SPEED = 1.0  # [m/s] sideways start velocity of the tip (linear profile, zero at the base)
MIN_TIP_DEFLECTION = 0.01  # [m]
MAX_TIP_REST_OFFSET = 1e-3  # [m] allowed distance of the tip from its start position at the end
FREQUENCY_REL_TOL = 0.15
DAMPING_RATIO_TOL = 0.02
SAG_REL_TOL = 0.05
HORIZONTAL_BASE = (0.3, 0.3, 0.5)  # [m] clamped end of the horizontal stem in the env frame
HORIZONTAL_ROT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))  # (x, y, z, w): +90 deg about y, stem along +x


@configclass
class StemSceneCfg(InteractiveSceneCfg):
    """Ground plane, one upright and one horizontal stem per env."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    stem: CableObjectCfg = stem_cfg().replace(prim_path="{ENV_REGEX_NS}/Stem")
    stem_horizontal: CableObjectCfg = stem_cfg().replace(
        prim_path="{ENV_REGEX_NS}/StemHorizontal",
        init_state=CableObjectCfg.InitialStateCfg(pos=HORIZONTAL_BASE, rot=HORIZONTAL_ROT),
    )


def _close(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= REL_TOL * abs(expected)


def _compare_gains(model_values, expected: dict[str, float]) -> tuple[bool, str]:
    """Check that the set of per-joint gains in the model equals the set of expected values."""
    model_values = {float(v) for v in model_values}
    ok = all(any(_close(v, e) for v in model_values) for e in expected.values()) and all(
        any(_close(v, e) for e in expected.values()) for v in model_values
    )
    model_info = sorted({f"{v:.4g}" for v in model_values}, key=float)
    expected_info = ", ".join(f"{name} = {value:.4g}" for name, value in expected.items())
    return ok, f"model {model_info}; expected {expected_info}"


def main() -> None:
    """Spawn the stem, compare the simulator state with the yaml and print a pass/fail summary."""
    params = stem_params()
    geometry, material, solver = params["geometry"], params["material"], params["solver"]
    num_segments = geometry["num_segments"]
    segment_length = geometry["length"] / num_segments
    area = math.pi * geometry["diameter"] ** 2 / 4
    area_moment = math.pi * geometry["diameter"] ** 4 / 64

    sim_cfg = sim_utils.SimulationCfg(
        dt=solver["sim_dt"],
        device=args_cli.device,
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=solver["vbd_iterations"]), num_substeps=solver["num_substeps"]
        ),
    )
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(1.5, 1.5, 0.8), target=(0.5, 0.0, 0.2))
        scene = InteractiveScene(StemSceneCfg(num_envs=args_cli.num_envs, env_spacing=ENV_SPACING))
        sim.reset()
        stem, stem_horizontal = scene["stem"], scene["stem_horizontal"]
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
        mass = float(model.body_mass.numpy().sum()) / (2 * args_cli.num_envs)  # two stems per env
        expected_mass = material["density"] * area * geometry["length"]
        results["mass"] = (_close(mass, expected_mass), f"{mass * 1e3:.3f} g (rho A L = {expected_mass * 1e3:.3f} g)")

        # null moduli: Newton uses the stretch stiffness for shear and the bend stiffness for twist
        shear_modulus = material["stretch_modulus"] if material["shear_modulus"] is None else material["shear_modulus"]
        bend_ke = material["bend_modulus"] * area_moment / segment_length
        twist_modulus = material["twist_modulus"]
        twist_ke = bend_ke if twist_modulus is None else twist_modulus * 2.0 * area_moment / segment_length
        expected_ke = {
            "stretch E A / l": material["stretch_modulus"] * area / segment_length,
            "shear G A / l": shear_modulus * area / segment_length,
            "bend E I / l": bend_ke,
            "twist G J / l": twist_ke,
        }
        results["joint stiffness"] = _compare_gains(model.joint_target_ke.numpy(), expected_ke)

        damping_time = material["damping_time"] or 0.0  # null -> no damping
        expected_kd = {"stretch, shear": 0.0, "tau * bend": damping_time * bend_ke, "tau * twist": damping_time * twist_ke}
        results["joint damping"] = _compare_gains(model.joint_target_kd.numpy(), expected_kd)

        # -- clamp the bases, kick the upright stem sideways, step
        fix_stem_base(stem)
        fix_stem_base(stem_horizontal)
        horizontal_start_poses = stem_horizontal.data.segment_pose_w.torch.clone()
        start_poses = stem.data.segment_pose_w.torch.clone()
        velocity = torch.zeros(args_cli.num_envs, num_segments, 6, device=sim.device)
        velocity[:, 1:, 0] = KICK_TIP_SPEED * torch.arange(1, num_segments, device=sim.device) / (num_segments - 1)
        stem.write_segment_velocity_to_sim_index(segment_velocity=velocity)

        base_error = torch.zeros(args_cli.num_envs, device=sim.device)
        tip_deflection = torch.zeros(args_cli.num_envs, device=sim.device)
        max_curvature = 0.0
        finite = True
        tip_x = []  # sideways tip position of the upright stem relative to its start, per step
        for _ in range(args_cli.steps):
            sim.step(render=False)
            scene.update(solver["sim_dt"])
            if sim.is_rendering:
                sim.render()
            poses = stem.data.segment_pose_w.torch
            finite = finite and bool(torch.isfinite(poses).all())
            base_error = torch.maximum(base_error, (poses[:, 0] - start_poses[:, 0]).abs().max(dim=-1).values)
            tip_deflection = torch.maximum(tip_deflection, (poses[:, -1, :2] - start_poses[:, -1, :2]).norm(dim=-1))
            tip_x.append((poses[:, -1, 0] - start_poses[:, -1, 0]).cpu())
            max_curvature = max(max_curvature, float(stem_geometry.joint_curvature(poses, segment_lengths).max()))

        results["base fixed"] = (
            float(base_error.max()) < 1e-6,
            f"max change of the base segment pose {float(base_error.max()):.1e} (position [m] / quaternion)",
        )
        results["deflects when kicked"] = (
            finite and float(tip_deflection.min()) > MIN_TIP_DEFLECTION,
            f"max tip deflection {[round(float(d), 4) for d in tip_deflection]} m, "
            f"max curvature {max_curvature:.2f} 1/m, all finite: {finite}",
        )
        tip_offset = (poses[:, -1, :3] - start_poses[:, -1, :3]).norm(dim=-1)
        results["springs back"] = (
            float(tip_offset.max()) < MAX_TIP_REST_OFFSET,
            f"tip offset from the start position after {args_cli.steps} steps "
            f"{[round(float(d) * 1e3, 3) for d in tip_offset]} mm",
        )

        # -- swing of the upright stem: frequency and damping ratio, per env
        free_length = geometry["length"] - segment_length  # the clamped segment does not bend
        expected_frequency = stem_reference.beam_first_frequency(
            free_length, geometry["diameter"], material["density"], material["bend_modulus"]
        )
        swings = [stem_reference.analyze_oscillation(x.tolist(), solver["sim_dt"]) for x in torch.stack(tip_x).T]
        frequencies = [frequency for _, frequency, _ in swings]
        damping_ratios = [damping_ratio for _, _, damping_ratio in swings]
        results["swing frequency"] = (
            all(abs(f - expected_frequency) <= FREQUENCY_REL_TOL * expected_frequency for f in frequencies),
            f"{[round(f, 2) for f in frequencies]} Hz (clamped beam of {free_length:.2f} m: {expected_frequency:.2f} Hz)",
        )
        expected_damping = [damping_time * math.pi * f for f in frequencies]
        results["damping ratio"] = (
            all(abs(z - e) <= DAMPING_RATIO_TOL for z, e in zip(damping_ratios, expected_damping)),
            f"{[round(z, 3) for z in damping_ratios]} (damping_time * pi * frequency = "
            f"{[round(e, 3) for e in expected_damping]})",
        )

        # -- horizontal stem: tip sag under its own weight
        horizontal_poses = stem_horizontal.data.segment_pose_w.torch
        tip_offset_in_segment = 0.5 * segment_length
        sag = (
            stem_geometry.point_pose(horizontal_start_poses, num_segments - 1, tip_offset_in_segment)[:, 2]
            - stem_geometry.point_pose(horizontal_poses, num_segments - 1, tip_offset_in_segment)[:, 2]
        )
        bend_sag, shear_sag = stem_reference.chain_tip_sag(
            geometry["length"],
            num_segments,
            geometry["diameter"],
            material["density"],
            material["bend_modulus"],
            shear_modulus,
        )
        expected_sag = bend_sag + shear_sag
        results["cantilever sag"] = (
            bool(((sag - expected_sag).abs() <= SAG_REL_TOL * expected_sag).all()),
            f"{[round(float(d) * 1e3, 2) for d in sag]} mm (by hand: {expected_sag * 1e3:.2f} mm = "
            f"bending {bend_sag * 1e3:.2f} + shear {shear_sag * 1e3:.2f})",
        )

        print(f"\n=== check_stem ({args_cli.num_envs} envs) ===")
        for name, (ok, info) in results.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        print("=== all passed ===" if all(ok for ok, _ in results.values()) else "=== FAILED ===")


if __name__ == "__main__":
    main()
