"""Accuracy of the stem model for one solver / stiffness setting (stem alone on Newton VBD, base clamped).

Runs one setting and prints one `RESULT` line; loop over settings in the shell to get a table. Defaults are the
values of `assets/stem/stem.yaml`, so without options it measures the setting the project uses.

Tests:
- `sag`: stem clamped horizontally, tip sag under its own weight after `--steps`, against the value computed
  by hand for the segment chain (bending + shear, `stem_reference.chain_tip_sag`). `ratio` = simulated / expected.
- `kick`: stem upright, tip kicked sideways with 1 m/s (linear velocity profile). Reports the first peak of the
  tip deflection, the oscillation frequency, the damping ratio (`stem_reference.analyze_oscillation`) and the tip
  offset at the end. For the placeholder stem expect roughly 35 mm, 5 Hz,
  a damping ratio of 0.05 (`damping_time` 3.2 ms) and a final offset near zero.

`cost` = substeps x iterations per simulation step. Results and trade-offs: docs/notes/2026-09-30.md.

With a viewer (`--viz newton_gl`) the test is paced to real time (`--slow_motion 5` = five times slower) and,
after the RESULT line, replayed from the start until the window is closed. The viewer's left panel has a
"Pause Simulation" / "Resume Simulation" button. Timing (`ms_per_step`) excludes rendering and pacing.

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/sweep_stem_solver.py --test sag
    uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick --viz newton_gl --slow_motion 5
    uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick --substeps 8 --iterations 20
    for r in 1 0.1 0.01 0.001; do
        uv run --extra isaacsim python scripts/sweep_stem_solver.py --test sag --shear_ratio $r | grep RESULT
    done
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Stem accuracy for one solver / stiffness setting.")
parser.add_argument("--test", choices=["sag", "kick"], default="sag", help="Which test to run.")
parser.add_argument("--substeps", type=int, default=None, help="Solver substeps per step (default: yaml).")
parser.add_argument("--iterations", type=int, default=None, help="VBD iterations per substep (default: yaml).")
parser.add_argument("--stretch_ratio", type=float, default=None, help="Stretch / bend modulus (default: yaml).")
parser.add_argument("--shear_ratio", type=float, default=None, help="Shear / bend modulus (default: yaml).")
parser.add_argument(
    "--damping",
    choices=["bend_twist", "all", "none"],
    default="bend_twist",
    help="Damped modes: bend and twist (as stem_cfg), all four (stretch and shear with tau * E A), or none.",
)
parser.add_argument("--steps", type=int, default=600, help="Simulation steps to run.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (for timing).")
parser.add_argument(
    "--slow_motion", type=float, default=1.0, help="With a viewer: play this many times slower than real time."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import math
import time

import torch
from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from stem_manip.assets.stem import fix_stem_base, stem_cfg, stem_params
from stem_manip.utils import stem_geometry, stem_reference

BASE_HEIGHT = 1.0  # [m] no ground plane in this scene
KICK_TIP_SPEED = 1.0  # [m/s]
WARMUP_STEPS = 20  # steps excluded from the timing


def _stem_cfg(params: dict) -> tuple[CableObjectCfg, float]:
    """Stem cfg with the command-line overrides applied. Returns the cfg and the shear modulus used [Pa]."""
    geometry, material = params["geometry"], params["material"]
    bend_modulus = material["bend_modulus"]
    area = math.pi * geometry["diameter"] ** 2 / 4
    segment_length = geometry["length"] / geometry["num_segments"]

    cfg = stem_cfg().replace(prim_path="{ENV_REGEX_NS}/Stem")
    cfg.init_state.pos = (0.0, 0.0, BASE_HEIGHT)
    if args_cli.test == "sag":
        cfg.spawn.positions = [(index * segment_length, 0.0, 0.0) for index in range(geometry["num_segments"] + 1)]

    physics_material = cfg.spawn.physics_material
    if args_cli.stretch_ratio is not None:
        physics_material.stretch_stiffness = args_cli.stretch_ratio * bend_modulus
    if args_cli.shear_ratio is not None:
        physics_material.shear_stiffness = args_cli.shear_ratio * bend_modulus
    if physics_material.shear_stiffness is None:  # Newton would fall back to the stretch stiffness
        physics_material.shear_stiffness = physics_material.stretch_stiffness

    if args_cli.damping == "none":
        physics_material.curves_bend_damping = None
        physics_material.curves_twist_damping = None
    elif args_cli.damping == "all":
        damping_time = material["damping_time"]
        physics_material.curves_stretch_damping = damping_time * physics_material.stretch_stiffness * area
        physics_material.curves_shear_damping = damping_time * physics_material.shear_stiffness * area
    return cfg, physics_material.shear_stiffness


def main() -> None:
    """Run one setting and print its RESULT line."""
    params = stem_params()
    geometry, material, solver = params["geometry"], params["material"], params["solver"]
    num_segments = geometry["num_segments"]
    segment_length = geometry["length"] / num_segments
    sim_dt = solver["sim_dt"]
    substeps = solver["num_substeps"] if args_cli.substeps is None else args_cli.substeps
    iterations = solver["vbd_iterations"] if args_cli.iterations is None else args_cli.iterations

    cfg, shear_modulus = _stem_cfg(params)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        stem: CableObjectCfg = cfg

    sim_cfg = sim_utils.SimulationCfg(
        dt=sim_dt,
        device=args_cli.device,
        physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=iterations), num_substeps=substeps),
    )
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(1.2, 1.2, BASE_HEIGHT + 0.4), target=(0.1, 0.0, BASE_HEIGHT + 0.1))
        scene = InteractiveScene(SceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0))
        sim.reset()
        stem = scene["stem"]
        fix_stem_base(stem)

        def start_test() -> None:
            """Put the stem into its start state and, for the kick test, give it the sideways velocity."""
            stem.write_segment_pose_to_sim_index(segment_pose=stem.data.default_segment_pose_w)
            velocity = torch.zeros(args_cli.num_envs, num_segments, 6, device=sim.device)
            if args_cli.test == "kick":
                profile = torch.arange(1, num_segments, device=sim.device) / (num_segments - 1)
                velocity[:, 1:, 0] = KICK_TIP_SPEED * profile
            stem.write_segment_velocity_to_sim_index(segment_velocity=velocity)

        def step() -> tuple[float, float, float]:
            """One simulation step. Returns the tip x (relative to the env) and z [m] and the solve time [s]."""
            step_start = time.perf_counter()
            sim.step(render=False)
            scene.update(sim_dt)
            tip = stem_geometry.point_pose(stem.data.segment_pose_w.torch, num_segments - 1, 0.5 * segment_length)[0]
            tip_x, tip_z = float(tip[0] - scene.env_origins[0, 0]), float(tip[2])
            solve_time = time.perf_counter() - step_start
            if sim.is_rendering:
                sim.render()
                time.sleep(max(0.0, args_cli.slow_motion * sim_dt - (time.perf_counter() - step_start)))
            return tip_x, tip_z, solve_time

        start_test()
        tip_x, tip_z, solve_times = [], [], []
        for _ in range(args_cli.steps):
            x, z, solve_time = step()
            tip_x.append(x)
            tip_z.append(z)
            solve_times.append(solve_time)
        ms_per_step = sum(solve_times[WARMUP_STEPS:]) / (args_cli.steps - WARMUP_STEPS) * 1e3

        stretch_ratio = cfg.spawn.physics_material.stretch_stiffness / material["bend_modulus"]
        setting = (
            f"test={args_cli.test} stretch_ratio={stretch_ratio:g} shear_ratio={shear_modulus / material['bend_modulus']:g}"
            f" damping={args_cli.damping} substeps={substeps} iterations={iterations} cost={substeps * iterations}"
        )
        if args_cli.test == "sag":
            sag_bend, sag_shear = stem_reference.chain_tip_sag(
                geometry["length"],
                num_segments,
                geometry["diameter"],
                material["density"],
                material["bend_modulus"],
                shear_modulus,
            )
            sag = BASE_HEIGHT - tip_z[-1]
            drift = tip_z[-101] - tip_z[-1] if len(tip_z) > 100 else math.nan
            outcome = (
                f"sag_mm={sag * 1e3:.2f} expected_mm={(sag_bend + sag_shear) * 1e3:.2f}"
                f" (bend {sag_bend * 1e3:.2f} + shear {sag_shear * 1e3:.2f}) ratio={sag / (sag_bend + sag_shear):.3f}"
                f" drift_last_100_steps_mm={drift * 1e3:.3f}"
            )
        else:
            peak, frequency, damping_ratio = stem_reference.analyze_oscillation(tip_x, sim_dt)
            outcome = (
                f"first_peak_mm={peak * 1e3:.1f} frequency_hz={frequency:.2f} damping_ratio={damping_ratio:.3f}"
                f" final_offset_mm={tip_x[-1] * 1e3:.2f}"
            )
        print(f"RESULT {setting} {outcome} ms_per_step={ms_per_step:.2f} (num_envs={args_cli.num_envs})")

        if sim.is_rendering:
            print("[INFO] Replaying until the viewer window is closed.")
            while sim.is_running():
                start_test()
                for _ in range(args_cli.steps):
                    if not sim.is_running():
                        break
                    step()


if __name__ == "__main__":
    main()
