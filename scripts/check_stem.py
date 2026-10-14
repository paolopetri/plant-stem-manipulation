"""Sanity check of the stem model `chain` alone (no robot), the model the training uses.

`cable` is kept in the repo for comparison but is not checked here (user, 2026-10-14).

Spawns `chain.stem_cfg()` in a few envs on a ground plane, and checks that the values of `stem_params("chain")`
(`assets/stem/stem.yaml` + `assets/stem/chain/chain.yaml`) arrive in the simulator:
- number of segments; start poses (segment centres on a vertical line above the base, tangent = local +Z up);
- start poses after a reset to the default state as Isaac Lab's `reset_scene_to_default` writes it (default root
  pose + env origin): the segment centres stay where they started (the root body's frame is the stem base);
- total mass against rho * A * L;
- per-joint stiffness against E I / l (bend) and G J / l (twist) (l = segment length);
- per-joint damping against damping_time * stiffness, and the joint armature.
The base is clamped by the fixed joint in the USD. The check kicks the upright stem (everything above the first
joint rotates rigidly, the centre of the last segment starts with 1 m/s sideways) and steps the simulation:
- the base segment keeps its start pose;
- the stem deflects (tip moves sideways by more than 1 cm) and all poses stay finite;
- the stem springs back (tip back within 1 mm of its start position at the end);
- swing frequency and damping ratio against the first bending mode of the exact chain with gravity
  (`stem_reference.chain_first_mode`, no armature: 5.11 Hz for the placeholder stem, as in
  docs/notes/2026-10-05.md, and a damping ratio of 0.053), within the tolerances below.
After the swing has died out, the last segment of the upright stem is pushed sideways with a constant force (0.05 N):
- deflects when pushed: deflection of the pushed point at rest within 10 % of the value computed by hand
  (`stem_reference.chain_push_deflection`; the formula ignores gravity, which adds about 4 %);
- returns after the push: the point is back within 1 mm after the force is removed.
A second stem per env is clamped horizontally and sags under its own weight:
- cantilever sag: tip drop at the end within 5 % of the value computed by hand for the segment chain
  (`stem_reference.chain_tip_sag`, bending only: the chain has no shear spring).
Solver settings come from the model (`physics_cfg()`, step `solver.sim_dt` and iterations from `chain.yaml`).

Tolerances of the swing (proposed in #70, decided once the user approves its PR). The reference is the exact chain
of the plant (no armature); the tolerance is the sum of the known differences between it and what this check
measures, each worked out before the run:
- frequency, 5 % (sum 4.4 %, rounded up):
  - joint armature 1e-4 kg m^2 (`chain.yaml`): -1.1 % (exact chain with armature: 5.055 Hz);
  - swing analysis (`analyze_oscillation`: zero crossings over two periods, the kick also excites the higher
    modes): -0.2 % (the exact chain's own kick response, analysed the same way: 5.101 Hz);
  - time step 2 ms: discretization error of mode 1 of order (omega dt)^2, at most (omega dt)^2 / 3 = 0.14 %
    (implicit Euler; omega dt = 0.064; with the TGS step h below it is smaller still);
  - PhysX's iterative solver may leave the chain softer: the push check accepts up to 6 % more deflection
    than the exact chain (10 % over the hand value, of which gravity takes 4 %: 9.44 / 9.09 mm), and the
    frequency goes with the square root of the stiffness: -3 %.
- damping ratio, +-0.007 (sum 0.0068):
  - joint armature: -0.0006 (exact chain with armature: 0.0528);
  - swing analysis (log decrement of the first two peaks): -0.0006 (exact chain's kick response: 0.0522
    against its modal 0.0528);
  - numerical damping of the implicit joint drives: PhysX's TGS solver integrates once per position iteration,
    h = 2 ms / 8, and an implicit Euler step damps mode 1 by about omega h / 2 = 0.004;
  - the solver softness above (-3 % in frequency) changes damping_time * spring / (2 omega * modal mass) by 3 %:
    0.0016.
Prints one `[PASS]` / `[FAIL]` line per check. Run it after every change to the stem model or its yaml.
`--env_spacing 0` stacks all envs at the world origin.

Usage (from the repo root; headless unless a visualizer is requested: `--viz kit` (Isaac Sim's window; the very
first start takes several minutes) or `--viz newton_gl`; `--slow_motion 5` plays five times slower than real time):
    uv run --extra isaacsim python scripts/check_stem.py
    uv run --extra isaacsim python scripts/check_stem.py --viz kit --slow_motion 5
    uv run --extra isaacsim python scripts/check_stem.py --viz newton_gl --slow_motion 5
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from stem_manip.assets.stem import stem_model, stem_params
from stem_manip.utils import check_report

parser = argparse.ArgumentParser(description="Sanity check of the stem model.")
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
FREQUENCY_REL_TOL = 0.05  # see the module docstring
DAMPING_RATIO_TOL = 0.007  # see the module docstring
SAG_REL_TOL = 0.05
RESET_SETTLE_TIME = 0.1  # [s] simulated time after the reset to default, before the poses are compared
RENDER_DT = 1.0 / 30.0  # [s] simulated time between rendered frames with a viewer
HORIZONTAL_BASE = (0.3, 0.3, 0.5)  # [m] clamped end of the horizontal stem in the env frame
HORIZONTAL_ROT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))  # (x, y, z, w): +90 deg about y, stem along +x

model_module = stem_model("chain")


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
    params = stem_params("chain")
    geometry, material, solver = params["geometry"], params["material"], params["solver"]
    num_segments = geometry["num_segments"]
    segment_length = geometry["length"] / num_segments
    area = math.pi * geometry["diameter"] ** 2 / 4
    area_moment = math.pi * geometry["diameter"] ** 4 / 64
    shear_modulus = math.inf  # no shear spring: the chain is rigid in shear
    sim_dt = solver["sim_dt"]

    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, physics=model_module.physics_cfg())
    with launch_simulation(sim_cfg, args_cli):
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # after the launch (Kit for PhysX)

        stem_cfg = model_module.stem_cfg()

        @configclass
        class StemSceneCfg(InteractiveSceneCfg):
            """Ground plane, light (for Kit's RTX viewer), one upright and one horizontal stem per env."""

            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            light = AssetBaseCfg(
                prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
            )
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
        expected_gains["armature"] = {"armature": params["joint"]["armature"]}
        gains = model_module.joint_gains(stem)
        for name, expected in expected_gains.items():
            results[f"joint {name}"] = _compare_gains(gains[name].ravel(), expected)

        # -- kick the upright stem sideways, step (the bases are clamped by the fixed joint in the USD)
        horizontal_start_poses = model_module.segment_poses(stem_horizontal).clone()
        set_force = model_module.segment_force_setter(stem)

        render_every = max(1, round(RENDER_DT / sim_dt))  # simulation steps per rendered frame

        def run(duration: float) -> None:
            """Step the simulation for `duration` simulated seconds; with a viewer, render and pace every frame."""
            frame_start = time.perf_counter()
            for step in range(round(duration / sim_dt)):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim_dt)
                if sim.is_rendering and step % render_every == 0:
                    sim.render()
                    frame_time = args_cli.slow_motion * render_every * sim_dt
                    time.sleep(max(0.0, frame_time - (time.perf_counter() - frame_start)))
                    frame_start = time.perf_counter()
                yield poses()

        # -- reset to the default state as Isaac Lab's `reset_scene_to_default` does (root pose = default root
        # pose + env origin, default joints); the stem must come back to its start poses, not move
        root_pose = stem.data.default_root_pose.torch.clone()
        root_pose[:, :3] += scene.env_origins
        stem.write_root_pose_to_sim_index(root_pose=root_pose)
        stem.write_root_velocity_to_sim_index(root_velocity=stem.data.default_root_vel.torch.clone())
        stem.write_joint_position_to_sim_index(position=stem.data.default_joint_pos.torch.clone())
        stem.write_joint_velocity_to_sim_index(velocity=stem.data.default_joint_vel.torch.clone())
        for _ in run(RESET_SETTLE_TIME):
            pass
        reset_err = float((poses()[..., :3] - expected_pos).norm(dim=-1).max())
        results["start poses after reset"] = (reset_err < POS_TOL, f"max centre error {reset_err:.1e} m")
        start_poses = poses().clone()

        model_module.write_kick(stem, KICK_TIP_SPEED / ((num_segments - 1.5) * segment_length))

        base_error = torch.zeros(args_cli.num_envs, device=sim.device)
        tip_deflection = torch.zeros(args_cli.num_envs, device=sim.device)
        max_curvature = 0.0
        finite = True
        tip_x = []  # sideways tip position of the upright stem relative to its start, per step
        for current in run(args_cli.kick_time):
            finite = finite and bool(torch.isfinite(current).all())
            base_error = torch.maximum(base_error, (current[:, 0] - start_poses[:, 0]).abs().max(dim=-1).values)
            tip_deflection = torch.maximum(tip_deflection, (current[:, -1, :2] - start_poses[:, -1, :2]).norm(dim=-1))
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

        # -- swing of the upright stem: frequency and damping ratio, per env, against mode 1 of the exact chain with
        # gravity and without armature (see the module docstring)
        expected_frequency, expected_damping = stem_reference.chain_first_mode(
            geometry["length"],
            num_segments,
            geometry["diameter"],
            material["density"],
            material["bend_modulus"],
            damping_time,
        )
        swings = [stem_reference.analyze_oscillation(x.tolist(), sim_dt) for x in torch.stack(tip_x).T]
        frequencies = [frequency for _, frequency, _ in swings]
        damping_ratios = [damping_ratio for _, _, damping_ratio in swings]
        results["swing frequency"] = (
            all(abs(f - expected_frequency) <= FREQUENCY_REL_TOL * expected_frequency for f in frequencies),
            f"{[round(f, 3) for f in frequencies]} Hz (exact chain with gravity: {expected_frequency:.3f} Hz "
            f"+- {FREQUENCY_REL_TOL:.0%})",
        )
        results["damping ratio"] = (
            all(abs(z - expected_damping) <= DAMPING_RATIO_TOL for z in damping_ratios),
            f"{[round(z, 4) for z in damping_ratios]} (exact chain with gravity: {expected_damping:.4f} "
            f"+- {DAMPING_RATIO_TOL})",
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
            f"{expected_deflection * 1e3:.2f} mm = bending {bend_deflection * 1e3:.2f} "
            f"+ shear {shear_deflection * 1e3:.2f})",
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

        print(f"\n=== check_stem, model chain ({args_cli.num_envs} envs) ===")
        check_report.finish(results)


if __name__ == "__main__":
    main()
