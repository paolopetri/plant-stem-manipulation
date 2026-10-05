"""Sanity check of a stem model alone (no robot). Model chosen with `--stem_model` (default `cable`).

Spawns the model's `stem_cfg()` in a few envs on a ground plane, and checks that the values of
`stem_params(model)` (`assets/stem/stem.yaml` + `assets/stem/<model>/<model>.yaml`) arrive in the simulator:
- number of segments; start poses (segment centres on a vertical line above the base, tangent = local +Z up);
- total mass against rho * A * L;
- per-joint stiffness against E I / l (bend), G J / l (twist) and, for the cable, E A / l (stretch, shear)
  (l = segment length);
- per-joint damping against damping_time * stiffness (bend and twist only), and the chain's joint armature.
Then clamps the base (`fix_stem_base`), kicks the upright stem (everything above the first joint rotates rigidly,
the centre of the last segment starts with 1 m/s sideways) and steps the simulation:
- the base segment keeps its start pose;
- the stem deflects (tip moves sideways) and all poses stay finite;
- the stem springs back (tip returns to its start position);
- swing frequency: within 15 % of the first bending frequency of a clamped beam. Its free length is the stem
  length minus half a segment: the first joint stands for the stem half a segment to either side of it. The
  simulated stem swings a little slower than this ideal beam (its own weight; the cable's soft shear spring);
- damping ratio: within 0.02 of damping_time * pi * frequency (the solvers add about 0.01-0.02 of their own).
After the swing has died out, the last segment of the upright stem is pushed sideways with a constant force:
- deflects when pushed: deflection of the pushed point at rest within 10 % of the value computed by hand
  (`stem_reference.chain_push_deflection`; the formula ignores gravity, which adds about 4 %);
- returns after the push: the point is back within 1 mm after the force is removed.
A second stem per env is clamped horizontally and sags under its own weight:
- cantilever sag: tip drop at the end within 5 % of the value computed by hand for the segment chain
  (bending + shear, `stem_reference.chain_tip_sag`; the chain has no shear spring, so bending only).
Solver settings come from the model (`physics_cfg()`, step `solver.sim_dt` from its yaml).
Cable only, at the end: the upright stem is at rest and compressed by its own weight:
- axial strain (reported, not checked): `joint_axial_strain` against the strain computed by hand
  (`stem_reference.chain_weight_strain`), for the envs nearest to and farthest from the world origin. At the
  current solver cost the stretch direction is not converged (docs/TODO.md -> M4).

Usage (from the repo root; headless unless a visualizer is requested with `--viz newton_gl`, which works for both
models; Kit's own viewer (`--viz kit`) hangs at start-up on this machine; `--slow_motion 5` plays five times slower
than real time):
    uv run --extra isaacsim python scripts/check_stem.py
    uv run --extra isaacsim python scripts/check_stem.py --stem_model chain
    uv run --extra isaacsim python scripts/check_stem.py --stem_model cable --viz newton_gl --slow_motion 5
    uv run --extra isaacsim python scripts/check_stem.py --stem_model chain --viz newton_gl --slow_motion 5
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from stem_manip.assets.stem import STEM_MODELS, stem_model, stem_params

parser = argparse.ArgumentParser(description="Sanity check of the stem model.")
parser.add_argument("--stem_model", choices=STEM_MODELS, default="cable", help="Stem model to check.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments.")
parser.add_argument("--kick_time", type=float, default=4.0, help="Simulated time after the kick [s].")
parser.add_argument("--env_spacing", type=float, default=1.0, help="Distance between neighbouring envs [m].")
parser.add_argument(
    "--slow_motion", type=float, default=1.0, help="With a viewer: play this many times slower than real time."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import math
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from stem_manip.utils import stem_geometry, stem_reference

POS_TOL = 1e-4  # [m]
REL_TOL = 1e-3
KICK_TIP_SPEED = 1.0  # [m/s] sideways start velocity of the centre of the last segment
MIN_TIP_DEFLECTION = 0.01  # [m]
MAX_TIP_REST_OFFSET = 1e-3  # [m] allowed distance of the tip from its start position at the end
PUSH_FORCE = 0.05  # [N] sideways force on the last segment of the upright stem
PUSH_TIME = 3.0  # [s] simulated time with the force, and again after releasing it
PUSH_REL_TOL = 0.10
FREQUENCY_REL_TOL = 0.15
DAMPING_RATIO_TOL = 0.02
SAG_REL_TOL = 0.05
HORIZONTAL_BASE = (0.3, 0.3, 0.5)  # [m] clamped end of the horizontal stem in the env frame
HORIZONTAL_ROT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))  # (x, y, z, w): +90 deg about y, stem along +x

model_module = stem_model(args_cli.stem_model)


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
    params = stem_params(args_cli.stem_model)
    geometry, material, solver = params["geometry"], params["material"], params["solver"]
    num_segments = geometry["num_segments"]
    segment_length = geometry["length"] / num_segments
    area = math.pi * geometry["diameter"] ** 2 / 4
    area_moment = math.pi * geometry["diameter"] ** 4 / 64
    has_stretch = "stretch_modulus" in material  # cable joints also have stretch and shear springs
    # shear modulus for the hand values; null: Newton uses the stretch stiffness; no shear spring: rigid in shear
    shear_modulus = (material["shear_modulus"] or material["stretch_modulus"]) if has_stretch else math.inf
    sim_dt = solver["sim_dt"]

    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, physics=model_module.physics_cfg())
    with launch_simulation(sim_cfg, args_cli):
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # after the launch (Kit for PhysX)

        stem_cfg = model_module.stem_cfg()

        @configclass
        class StemSceneCfg(InteractiveSceneCfg):
            """Ground plane, one upright and one horizontal stem per env."""

            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            stem = stem_cfg.replace(prim_path="{ENV_REGEX_NS}/Stem")
            stem_horizontal = stem_cfg.replace(
                prim_path="{ENV_REGEX_NS}/StemHorizontal",
                init_state=stem_cfg.init_state.replace(pos=HORIZONTAL_BASE, rot=HORIZONTAL_ROT),
            )

        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(1.5, 1.5, 0.8), target=(0.5, 0.0, 0.2))
        scene = InteractiveScene(StemSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing))
        sim.reset()
        stem, stem_horizontal = scene["stem"], scene["stem_horizontal"]
        segment_lengths = torch.full((num_segments,), segment_length, device=sim.device)
        results = {}

        def poses() -> torch.Tensor:
            return model_module.segment_poses(stem)

        # -- start state
        start_poses = poses().clone()
        results["segments"] = (
            start_poses.shape[1] == num_segments,
            f"{start_poses.shape[1]} (yaml {num_segments})",
        )

        base = scene.env_origins + torch.tensor(geometry["base_position"], device=sim.device)
        expected_pos = base.unsqueeze(1).repeat(1, num_segments, 1)
        expected_pos[..., 2] += (torch.arange(num_segments, device=sim.device) + 0.5) * segment_length
        pos_err = float((start_poses[..., :3] - expected_pos).norm(dim=-1).max())
        up = torch.zeros_like(start_poses[..., :3])
        up[..., 2] = 1.0
        tangent_err = float((quat_apply(start_poses[..., 3:], up) - up).norm(dim=-1).max())
        results["start poses"] = (
            pos_err < POS_TOL and tangent_err < 1e-4,
            f"max centre error {pos_err:.1e} m, max tangent error {tangent_err:.1e}",
        )

        # -- parameters in the simulation
        mass = model_module.segment_masses(stem).sum(dim=-1)
        expected_mass = material["density"] * area * geometry["length"]
        results["mass"] = (
            all(_close(float(m), expected_mass) for m in mass),
            f"{float(mass.max()) * 1e3:.3f} g (rho A L = {expected_mass * 1e3:.3f} g)",
        )

        bend_ke = material["bend_modulus"] * area_moment / segment_length
        twist_modulus = material["twist_modulus"]
        twist_ke = bend_ke if twist_modulus is None else twist_modulus * 2.0 * area_moment / segment_length
        damping_time = material["damping_time"] or 0.0  # null -> no damping
        expected_gains = {
            "stiffness": {"bend E I / l": bend_ke, "twist G J / l": twist_ke},
            "damping": {"tau * bend": damping_time * bend_ke, "tau * twist": damping_time * twist_ke},
        }
        if has_stretch:
            expected_gains["stiffness"] |= {
                "stretch E A / l": material["stretch_modulus"] * area / segment_length,
                "shear G A / l": shear_modulus * area / segment_length,
            }
            expected_gains["damping"] |= {"stretch, shear": 0.0}
        if "joint" in params:
            expected_gains["armature"] = {"armature": params["joint"]["armature"]}
        gains = model_module.joint_gains(stem)
        for name, expected in expected_gains.items():
            results[f"joint {name}"] = _compare_gains(gains[name].ravel(), expected)

        # -- clamp the bases, kick the upright stem sideways, step
        model_module.fix_stem_base(stem)
        model_module.fix_stem_base(stem_horizontal)
        horizontal_start_poses = model_module.segment_poses(stem_horizontal).clone()
        set_force = model_module.segment_force_setter(stem)  # before the first step (cable)

        def run(duration: float) -> None:
            """Step the simulation for `duration` simulated seconds, paced for the viewer."""
            for _ in range(round(duration / sim_dt)):
                step_start = time.perf_counter()
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim_dt)
                if sim.is_rendering:
                    sim.render()
                    time.sleep(max(0.0, args_cli.slow_motion * sim_dt - (time.perf_counter() - step_start)))
                yield poses()

        model_module.write_kick(stem, KICK_TIP_SPEED / ((num_segments - 1.5) * segment_length))

        base_error = torch.zeros(args_cli.num_envs, device=sim.device)
        tip_deflection = torch.zeros(args_cli.num_envs, device=sim.device)
        max_curvature = 0.0
        finite = True
        tip_x = []  # sideways tip position of the upright stem relative to its start, per step
        for current in run(args_cli.kick_time):
            finite = finite and bool(torch.isfinite(current).all())
            base_error = torch.maximum(base_error, (current[:, 0] - start_poses[:, 0]).abs().max(dim=-1).values)
            tip_deflection = torch.maximum(
                tip_deflection, (current[:, -1, :2] - start_poses[:, -1, :2]).norm(dim=-1)
            )
            tip_x.append((current[:, -1, 0] - start_poses[:, -1, 0]).cpu())
            max_curvature = max(max_curvature, float(stem_geometry.joint_curvature(current, segment_lengths).max()))

        results["base fixed"] = (
            float(base_error.max()) < 1e-6,
            f"max change of the base segment pose {float(base_error.max()):.1e} (position [m] / quaternion)",
        )
        results["deflects when kicked"] = (
            finite and float(tip_deflection.min()) > MIN_TIP_DEFLECTION,
            f"max tip deflection {[round(float(d), 4) for d in tip_deflection]} m, "
            f"max curvature {max_curvature:.2f} 1/m, all finite: {finite}",
        )
        tip_offset = (poses()[:, -1, :3] - start_poses[:, -1, :3]).norm(dim=-1)
        results["springs back"] = (
            float(tip_offset.max()) < MAX_TIP_REST_OFFSET,
            f"tip offset from the start position after {args_cli.kick_time} s "
            f"{[round(float(d) * 1e3, 3) for d in tip_offset]} mm",
        )

        # -- swing of the upright stem: frequency and damping ratio, per env
        free_length = geometry["length"] - 0.5 * segment_length  # see the module docstring
        expected_frequency = stem_reference.beam_first_frequency(
            free_length, geometry["diameter"], material["density"], material["bend_modulus"]
        )
        swings = [stem_reference.analyze_oscillation(x.tolist(), sim_dt) for x in torch.stack(tip_x).T]
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

        # -- push the last segment of the upright stem sideways (+x) with a constant force, then release it
        push = torch.zeros(args_cli.num_envs, 3, device=sim.device)
        push[:, 0] = PUSH_FORCE
        tip_before_push = poses()[:, -1, :3].clone()
        set_force(num_segments - 1, push)
        for _ in run(PUSH_TIME):
            pass
        push_deflection = poses()[:, -1, 0] - tip_before_push[:, 0]
        set_force(num_segments - 1, None)
        for _ in run(PUSH_TIME):
            pass
        push_rest_offset = (poses()[:, -1, :3] - tip_before_push).norm(dim=-1)

        bend_deflection, shear_deflection = stem_reference.chain_push_deflection(
            PUSH_FORCE, geometry["length"], num_segments, geometry["diameter"], material["bend_modulus"], shear_modulus
        )
        expected_deflection = bend_deflection + shear_deflection
        results["deflects when pushed"] = (
            bool(((push_deflection - expected_deflection).abs() <= PUSH_REL_TOL * expected_deflection).all()),
            f"{[round(float(d) * 1e3, 2) for d in push_deflection]} mm with {PUSH_FORCE} N (by hand, without gravity: "
            f"{expected_deflection * 1e3:.2f} mm = bending {bend_deflection * 1e3:.2f} + shear {shear_deflection * 1e3:.2f})",
        )
        results["returns after the push"] = (
            float(push_rest_offset.max()) < MAX_TIP_REST_OFFSET,
            f"offset {[round(float(d) * 1e3, 3) for d in push_rest_offset]} mm after releasing the force",
        )

        # -- horizontal stem: tip sag under its own weight
        horizontal_poses = model_module.segment_poses(stem_horizontal)
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

        # -- cable: axial strain of the upright stem at rest, against the hand value
        if has_stretch:
            strain = stem_geometry.joint_axial_strain(poses(), segment_lengths)
            expected_strain = torch.tensor(
                stem_reference.chain_weight_strain(
                    geometry["length"], num_segments, material["density"], material["stretch_modulus"]
                ),
                device=sim.device,
            )
            strain_error = (strain - expected_strain).abs().max(dim=-1).values
            distance = scene.env_origins[:, :2].norm(dim=-1)
            nearest, farthest = int(distance.argmin()), int(distance.argmax())
            stretch_stiffness = material["stretch_modulus"] * area  # E A [N]: strain error -> force error
            for label, env in (("nearest", nearest), ("farthest", farthest)):
                print(
                    f"[INFO] axial strain, {label} env ({float(distance[env]):.1f} m from the origin): base joint "
                    f"{float(strain[env, 0]):.2e} (by hand {float(expected_strain[0]):.2e}), max error "
                    f"{float(strain_error[env]):.1e} (= {float(strain_error[env]) * stretch_stiffness:.4f} N)"
                )

        print(f"\n=== check_stem, model {args_cli.stem_model} ({args_cli.num_envs} envs) ===")
        for name, (ok, info) in results.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        print("=== all passed ===" if all(ok for ok, _ in results.values()) else "=== FAILED ===")


if __name__ == "__main__":
    main()
