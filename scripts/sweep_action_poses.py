"""Sweep: does the push task's action follow its steps everywhere in the workspace (free space, no stem)?

Uses the env of `StemManip-Push-Position-FR3-v0` itself (its action term: Franka's Cartesian impedance law with an
integrated target, speed cap, step-change limit and target-offset clamp), with the stem moved out of reach behind the
robot. Each env first drives the tool tip to its own start point (the 8 corners of the box below, plus the env's start
pose; step-change limit and clamp switched off for this) and holds. Two modes:
- default: translation moves (`--steps` full steps along +x, -x, +y, -y, +z, -z, and reversals +x then -x etc., whose
  second segment is twice as long so that the target brakes *and* reverses) and rotation moves (`--rot_steps`, about
  the same base axes), each followed by a hold, one after the other. With the step-change limit the target ramps up
  at the start of a move and brakes during the hold; at the default 20 steps and the cfg's 0.08 mm step change it
  reaches 1.6 mm per step (5 cm/s), so this mode tests the acceleration limit in many directions, not the speed caps.
- `--full_speed`: one straight move per axis (translation, then rotation) that ramps up to the cap, stays there for
  `FULL_SPEED_STEPS` policy steps and brakes to rest in the hold (at the cfg defaults about 16 cm / 120 deg); each
  env goes back to its start point before every move. Directions with room for this: x and y towards the middle of
  the box, z up (down from the high start points would come close to the ground), rotations positive.
Measured per start point and move, against the criteria of docs/overleaf_folder/open_questions/
action_limits_problem.tex (decisions 2026-10-07):
- following error: largest distance to the target over the whole move and the hold (<= 2 mm);
- overshoot past the target's final position, along the last direction (<= 5 mm); end error after the hold (<= 1 mm);
- the fork's tilt from its orientation at the start of the move (<= 1 deg).
Rotation moves: following error (angle to the target orientation, <= 2 deg), overshoot past the target's final
orientation about the last axis (<= 2 deg), tool-tip drift while turning (the position target does not move, <= 2 mm);
the end error is reported.
Joint limits (user, 2026-10-08): there the arm, not the controller, limits the motion, so these moves are reported but
not counted in pass/fail: default mode, start points within 0.25 rad of a joint limit after settling ("near joint
limit, not counted"); `--full_speed`, moves that come within 0.25 rad of a joint limit anywhere (the count of counted
moves is printed per start point).

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/sweep_action_poses.py
    uv run --extra isaacsim python scripts/sweep_action_poses.py --full_speed
    uv run --extra isaacsim python scripts/sweep_action_poses.py --max_step 0.0025 --damping franka
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Action following over the workspace (free space).")
parser.add_argument("--steps", type=int, default=20, help="Policy steps per translation segment (default mode).")
parser.add_argument(
    "--rot_steps", type=int, default=20, help="Policy steps per rotation segment (default mode; like --steps)."
)
parser.add_argument("--hold_steps", type=int, default=45, help="Policy steps of zero action after each move.")
parser.add_argument("--full_speed", action="store_true", help="One move per axis up to the speed caps (see above).")
parser.add_argument("--max_step", type=float, default=None, help="Override the action's max_step [m].")
parser.add_argument("--max_rot_step", type=float, default=None, help="Override the action's max_rot_step [rad].")
parser.add_argument("--max_step_change", type=float, default=None, help="Override max_step_change [m].")
parser.add_argument("--max_rot_step_change", type=float, default=None, help="Override max_rot_step_change [rad].")
parser.add_argument("--max_target_offset", type=float, default=None, help="Override max_target_offset [m].")
parser.add_argument("--damping", choices=["franka", "apparent_mass"], default=None, help="Override the damping.")
parser.add_argument("--pos_stiffness", type=float, default=None, help="Override K_p [N/m].")
parser.add_argument("--rot_stiffness", type=float, default=None, help="Override K_o [N m/rad].")
parser.add_argument("--nullspace_stiffness", type=float, default=None, help="Override k_ns [N m/rad].")
add_launcher_args(parser)
args_cli = parser.parse_args()

import itertools
import math

import gymnasium as gym
import torch

import stem_manip.tasks  # noqa: F401  (registers the task)
from stem_manip.tasks.push_position.env_cfg import StemPushPositionEnvCfg
from stem_manip.utils.impedance import rotvec_between

TASK = "StemManip-Push-Position-FR3-v0"
BOX_X, BOX_Y, BOX_Z = (0.35, 0.60), (-0.20, 0.20), (0.20, 0.45)  # [m] tool-tip start points, robot base frame
BOX_CENTER = (0.475, 0.0, 0.325)  # [m] full-speed moves in x and y head towards it
START_POSE = (0.40, 0.0, 0.50)  # [m] the env's start pose
STEM_AWAY = (-0.8, 0.8, 0.0)  # [m] stem base, out of the fork's reach behind the robot
FOLLOW_TOL, OVERSHOOT_TOL, END_TOL, TILT_TOL = 2e-3, 5e-3, 1e-3, math.radians(1.0)
ROT_FOLLOW_TOL, ROT_OVERSHOOT_TOL, ROT_DRIFT_TOL = math.radians(2.0), math.radians(2.0), 2e-3  # user, 2026-10-07
NEAR_LIMIT = 0.25  # [rad] closer to a joint limit, a move is not counted (user, 2026-10-08)
FULL_SPEED_STEPS = 10  # policy steps at the cap in a full-speed move
SETTLE_STEPS = 20  # extra hold after a full-speed move has braked to rest
MAX_DRIVE_STEPS = 2000  # go_to_starts gives up after this many policy steps (the drives take ~150)
AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
# default-mode moves: name -> segments (direction, length in units of --steps / --rot_steps)
MOVES = {
    **{
        f"{sign}{a}": [(tuple(sign_v * c for c in d), 1)]
        for a, d in AXES.items()
        for sign, sign_v in (("+", 1), ("-", -1))
    },
    **{f"+{a}-{a}": [(d, 1), (tuple(-c for c in d), 2)] for a, d in AXES.items()},
}


def main() -> None:
    """Run all start points in parallel envs and print a table plus one RESULT line per start point."""
    starts = [*itertools.product(BOX_X, BOX_Y, BOX_Z), START_POSE]
    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = len(starts)
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.events.spawn_stem = None  # no spawn area: the stem stays at STEM_AWAY
    env_cfg.scene.stem.init_state.pos = STEM_AWAY
    env_cfg.episode_length_s = 1e4  # no time-out during the sweep
    # no joint-margin reset: the sweep reports moves near a joint limit itself (a reset start point would never reach
    # its start, and go_to_starts would loop forever)
    env_cfg.terminations.joint_margin = None
    action_cfg = env_cfg.actions.tool_tip
    for name, value in (
        ("max_step", args_cli.max_step),
        ("max_rot_step", args_cli.max_rot_step),
        ("damping", args_cli.damping),
        ("max_step_change", args_cli.max_step_change),
        ("max_rot_step_change", args_cli.max_rot_step_change),
        ("max_target_offset", args_cli.max_target_offset),
        ("stiffness_pos", args_cli.pos_stiffness),
        ("stiffness_rot", args_cli.rot_stiffness),
        ("stiffness_nullspace", args_cli.nullspace_stiffness),
    ):
        if value is not None:
            setattr(action_cfg, name, value)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        device, n = env.device, env.num_envs
        term = env.action_manager.get_term("tool_tip")
        cfg = term.cfg
        start_t = torch.tensor(starts, device=device)
        start_quat = term.target_quat().clone()  # the start pose's orientation, kept at every start point
        robot = env.scene["robot"]
        limits = robot.data.joint_pos_limits.torch[:, :7]

        def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Angle between orientations a and b (n, 4) [rad]."""
            return rotvec_between(a, b).norm(dim=-1)

        def joint_margin() -> tuple[torch.Tensor, torch.Tensor]:
            """Smallest distance to a joint limit (n,) [rad] and the joint (n,) it belongs to."""
            q = robot.data.joint_pos.torch[:, :7]
            return torch.minimum(q - limits[..., 0], limits[..., 1] - q).min(dim=-1)

        def go_to_starts() -> None:
            """Drive every env's target to its start pose (limiter and clamp off), then hold until at rest."""
            keys = ("max_step_change", "max_rot_step_change", "max_target_offset", "max_target_rot_offset")
            saved = {k: getattr(cfg, k) for k in keys}
            for k in keys:
                setattr(cfg, k, 1e3)
            for _ in range(MAX_DRIVE_STEPS):
                remaining = start_t - term.target()
                remaining_rot = rotvec_between(start_quat, term.target_quat())
                if float(remaining.norm(dim=-1).max()) < 1e-6 and float(remaining_rot.norm(dim=-1).max()) < 1e-6:
                    break
                env.step(
                    torch.cat(((remaining / cfg.max_step), (remaining_rot / cfg.max_rot_step)), dim=-1).clamp(-1, 1)
                )
            else:
                missing = ((remaining.norm(dim=-1) >= 1e-6) | (remaining_rot.norm(dim=-1) >= 1e-6)).nonzero().flatten()
                raise RuntimeError(
                    f"go_to_starts: envs {missing.tolist()} did not reach their start point in {MAX_DRIVE_STEPS} "
                    "policy steps (reset by a termination?)"
                )
            for _ in range(2 * args_cli.hold_steps):
                env.step(torch.zeros(n, 6, device=device))
            for k, v in saved.items():
                setattr(cfg, k, v)

        def run_move(segments: list[tuple[torch.Tensor, int]], hold: int, rotation: bool) -> dict[str, torch.Tensor]:
            """Full action along per-env directions (n, 3) for a number of policy steps each, then a zero hold.

            Returns per env (n,): largest following error, overshoot past the final target along the last direction,
            end error, the tilt (translation) or tool-tip drift (rotation), and the smallest joint margin.
            """
            pos0, quat0 = term.tool_pose()[0].clone(), term.target_quat().clone()
            follow, side = torch.zeros(n, device=device), torch.zeros(n, device=device)
            margin = torch.full((n,), math.inf, device=device)
            held = []  # tool positions (orientations) during the hold
            schedule = [d for d, k in segments for _ in range(k)] + [None] * hold
            for d in schedule:
                action = torch.zeros(n, 6, device=device)
                if d is not None:
                    action[:, slice(3, 6) if rotation else slice(0, 3)] = d
                env.step(action)
                pos, quat = term.tool_pose()
                if rotation:
                    follow = torch.maximum(follow, angle_between(quat, term.target_quat()))
                    side = torch.maximum(side, (pos - pos0).norm(dim=-1))
                else:
                    follow = torch.maximum(follow, (pos - term.target()).norm(dim=-1))
                    side = torch.maximum(side, angle_between(quat, quat0))
                margin = torch.minimum(margin, joint_margin()[0])
                if d is None:
                    held.append((quat if rotation else pos).clone())
            last = segments[-1][0]
            if rotation:
                end = term.target_quat().clone()
                overshoot = torch.stack([(rotvec_between(q, end) * last).sum(-1) for q in held]).max(dim=0).values
                end_error = angle_between(term.tool_pose()[1], end)
            else:
                end = term.target().clone()
                overshoot = torch.stack([((p - end) * last).sum(-1) for p in held]).max(dim=0).values
                end_error = (term.tool_pose()[0] - end).norm(dim=-1)
            return {
                "follow": follow,
                "overshoot": overshoot.clamp(min=0.0),
                "end": end_error,
                "drift" if rotation else "tilt": side,
                "margin": margin,
            }

        go_to_starts()
        settle = (term.tool_pose()[0] - start_t).norm(dim=-1)
        margin_min, margin_joint = joint_margin()

        def unit(direction: tuple[int, int, int]) -> torch.Tensor:
            return torch.tensor(direction, device=device, dtype=torch.float32).expand(n, 3)

        results, rot_results = {}, {}  # move -> dict of (n,) tensors, incl. "counted"
        if args_cli.full_speed:
            ramp = math.ceil(cfg.max_step / cfg.max_step_change) + FULL_SPEED_STEPS
            rot_ramp = math.ceil(cfg.max_rot_step / cfg.max_rot_step_change) + FULL_SPEED_STEPS
            towards = torch.where(start_t <= torch.tensor(BOX_CENTER, device=device), 1.0, -1.0)
            for i, a in enumerate(AXES):
                direction = unit(AXES[a]) * (towards[:, i : i + 1] if a != "z" else 1.0)
                go_to_starts()
                results[a] = run_move([(direction, ramp)], ramp + SETTLE_STEPS, rotation=False)
            for a in AXES:
                go_to_starts()
                rot_results[f"r{a}"] = run_move([(unit(AXES[a]), rot_ramp)], rot_ramp + SETTLE_STEPS, rotation=True)
            for r in (*results.values(), *rot_results.values()):
                r["counted"] = r["margin"] >= NEAR_LIMIT
        else:
            for name, segments in MOVES.items():
                results[name] = run_move(
                    [(unit(d), k * args_cli.steps) for d, k in segments], args_cli.hold_steps, False
                )
            for name, segments in MOVES.items():
                rot_results[f"r{name}"] = run_move(
                    [(unit(d), k * args_cli.rot_steps) for d, k in segments], args_cli.hold_steps, True
                )
            for r in (*results.values(), *rot_results.values()):
                r["counted"] = margin_min >= NEAR_LIMIT

        def worst(table: dict, key: str, i: int) -> tuple[float, str]:
            """Worst value of `key` over the counted moves of start point i (over all moves if none is counted, for
            the report), and the move it occurred in."""
            names = [k for k in table if bool(table[k]["counted"][i])] or list(table)
            name = max(names, key=lambda k: float(table[k][key][i]))
            return float(table[name][key][i]), name

        def any_counted(table: dict, i: int) -> bool:
            return any(bool(r["counted"][i]) for r in table.values())

        def verdict(table: dict, i: int, ok: bool) -> str:
            counted = sum(bool(r["counted"][i]) for r in table.values())
            if counted == 0:
                return "near joint limit, not counted"
            label = "PASS" if ok else "FAIL"
            return f"{label} ({counted}/{len(table)} moves counted)" if args_cli.full_speed else label

        mode = (
            "full speed: one move per axis up to the caps"
            if args_cli.full_speed
            else (f"{args_cli.steps} / {args_cli.rot_steps} full steps per segment")
        )
        print(
            f"\n=== sweep_action_poses ({n} start points, {mode}; damping {cfg.damping}, K_p {cfg.stiffness_pos}, "
            f"K_o {cfg.stiffness_rot}, max_step {cfg.max_step * 1e3:.2f} mm = {cfg.max_step / env.step_dt * 100:.2f} "
            f"cm/s, max_rot_step {math.degrees(cfg.max_rot_step):.2f} deg = "
            f"{math.degrees(cfg.max_rot_step) / env.step_dt:.0f} deg/s, step change "
            f"{cfg.max_step_change * 1e3:.2f} mm / {math.degrees(cfg.max_rot_step_change):.2f} deg, target offset "
            f"{cfg.max_target_offset * 1e3:.1f} mm) ==="
        )
        print(f"{'start point [m]':<22} {'settle':>7} {'follow':>14} {'overshoot':>14} {'end':>7} {'tilt':>12}")
        print(f"{'':<22} {'mm':>7} {'mm (worst dir)':>14} {'mm (worst dir)':>14} {'mm':>7} {'deg (dir)':>12}")
        all_ok = True
        for i, start in enumerate(starts):
            follow, follow_dir = worst(results, "follow", i)
            overshoot, overshoot_dir = worst(results, "overshoot", i)
            end, _ = worst(results, "end", i)
            peak_tilt, tilt_dir = worst(results, "tilt", i)
            ok = follow <= FOLLOW_TOL and overshoot <= OVERSHOOT_TOL and end <= END_TOL and peak_tilt <= TILT_TOL
            all_ok = all_ok and (ok or not any_counted(results, i))
            label = "(" + ", ".join(f"{v:.2f}" for v in start) + ")"
            print(
                f"{label:<22} {float(settle[i]) * 1e3:7.2f} {follow * 1e3:8.2f} ({follow_dir}) "
                f"{overshoot * 1e3:8.2f} ({overshoot_dir}) {end * 1e3:7.2f} {math.degrees(peak_tilt):7.2f} ({tilt_dir})"
                f"  limit margin {float(margin_min[i]):.2f} rad (joint {int(margin_joint[i]) + 1})  "
                f"{verdict(results, i, ok)}"
            )
            print(
                f"RESULT start={label} follow_mm={follow * 1e3:.2f} overshoot_mm={overshoot * 1e3:.2f} "
                f"end_mm={end * 1e3:.2f} tilt_deg={math.degrees(peak_tilt):.2f} ok={ok}"
            )
        print(
            "--- rotations: "
            + (
                "one move per axis up to the cap"
                if args_cli.full_speed
                else f"{args_cli.rot_steps} full steps per segment about the base axes"
            )
            + " ---"
        )
        print(f"{'start point [m]':<22} {'follow':>14} {'overshoot':>14} {'end':>7} {'tip drift':>14}")
        print(f"{'':<22} {'deg (dir)':>14} {'deg (dir)':>14} {'deg':>7} {'mm (dir)':>14}")
        for i, start in enumerate(starts):
            follow, follow_dir = worst(rot_results, "follow", i)
            overshoot, overshoot_dir = worst(rot_results, "overshoot", i)
            end, _ = worst(rot_results, "end", i)
            drift, drift_dir = worst(rot_results, "drift", i)
            label = "(" + ", ".join(f"{v:.2f}" for v in start) + ")"
            ok = follow <= ROT_FOLLOW_TOL and overshoot <= ROT_OVERSHOOT_TOL and drift <= ROT_DRIFT_TOL
            all_ok = all_ok and (ok or not any_counted(rot_results, i))
            print(
                f"{label:<22} {math.degrees(follow):8.2f} ({follow_dir}) {math.degrees(overshoot):8.2f} "
                f"({overshoot_dir}) {math.degrees(end):7.2f} {drift * 1e3:8.2f} ({drift_dir})  "
                f"{verdict(rot_results, i, ok)}"
            )
            print(
                f"RESULT_ROT start={label} follow_deg={math.degrees(follow):.2f} overshoot_deg="
                f"{math.degrees(overshoot):.2f} end_deg={math.degrees(end):.2f} drift_mm={drift * 1e3:.2f} ok={ok}"
            )
        print("=== all passed ===" if all_ok else "=== FAILED ===")
        env.close()


if __name__ == "__main__":
    main()
