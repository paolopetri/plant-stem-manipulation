"""Env check for stage 1 (`StemManip-Push-Position-FR3-v0`): the reward and termination terms (M4 step 3).

Every reward and termination term is wrapped so that, when the managers evaluate it (before the env's automatic
reset), its value is recorded together with values computed here independently of the MDP code:
- stem tip from the segment poses (segment 19, +L/2) and the command's target -> distance d and height error dz;
- tool tip from the action term's pose -> distance to the stem, sampled at 1 mm along the arc length;
- curvature per joint from the segment poses; contact force from the sensor's history (largest single
  segment-body contact per physics step, mean over the policy step); joint margin from the joint
  positions and limits.
Decided values (user, 2026-10-09), fixed here, not read from the cfg: std 0.05 / 0.01 m (distance), 0.003 m (height),
0.1 m (approach); curvature penalty above 0.8 x 5 1/m, limit 5 1/m; contact penalty above 2 N, limit 5 N; joint
margin 0.25 rad; episode 10 s.

Phases (the stem base fixed at x 0.65 m, y 0, except in `rest`):
- rest: zero action for a whole episode (spawn area as in training). Time out fires exactly at the last step and
  nothing else fires before; curvature and contact penalties 0, contact force 0, action rate 0.
- action_rate: alternating actions; the term equals sum((a_t - a_t-1)^2) of the actions sent.
- push: drive the fork behind the stem at 0.30 m height (forward first, then down; the drive must reach its goal
  within 2 mm without a reset), at full speed until 1 cm before contact, then push it slowly (5 cm/s) 4 cm into the
  slot. Approach
  reward rises and is > 0.9 in contact; contact force > 0.05 N, below 2 N, penalty 0. Then, as a mechanism check with
  lowered thresholds (not the decided values): penalty threshold 0 N -> penalty = F^2; limit at half the current force
  -> `contact_force_limit` fires in every env.
- impact: the fork at full speed into the stem at 0.25 m height (informational: the largest mean force of a policy
  step; whether the 5 N limit is reached by an impact; fails only if the fork does not reach / touch the stem).
- curvature: a force on the tip segment, ramped 0 -> 2 N; the penalty becomes > 0 above 4 1/m and
  `curvature_limit` fires (in every env, at the same step) above 5 1/m; nothing fires earlier.
- joint_margin: the fork moved straight down at full speed from the start pose; `joint_margin` fires in every env.
Only `rest` runs on the decided 10 s episode; the scripted phases take longer and run without a time out, and a
reset anywhere except where a phase makes a term fire fails the check. In every phase and at every step: each term
equals its formula of the independent values (tolerance 1e-4 for the rewards, 0.5 mm on the approach distance), each
termination fires exactly when its independent value crosses the limit, `terminated` = any early termination, metric
`height_error` = dz.

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/check_push_terms.py
    uv run --extra isaacsim python scripts/check_push_terms.py --num_envs 4 --phases push --viz kit --slow_motion 3
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

PHASES = ("rest", "action_rate", "push", "impact", "curvature", "joint_margin")
parser = argparse.ArgumentParser(description="Reward and termination check for the stage-1 push task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs.")
parser.add_argument("--phases", nargs="+", default=list(PHASES), choices=PHASES, help="Phases to run, in order.")
parser.add_argument(
    "--slow_motion", type=float, default=1.0, help="With a viewer: play this many times slower than real time."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import math
import time

import gymnasium as gym
import torch

from isaaclab.utils.math import subtract_frame_transforms

import stem_manip.tasks  # noqa: F401  (registers the task)
from stem_manip.assets.stem import stem_model, stem_params
from stem_manip.tasks.push_position.env_cfg import STEM_MODEL, StemPushPositionEnvCfg
from stem_manip.utils.stem_geometry import joint_curvature, point_pose, stem_points

TASK = "StemManip-Push-Position-FR3-v0"
# decided values (user, 2026-10-09)
STD = {"distance_coarse": 0.05, "distance_fine": 0.01, "height": 0.003, "approach": 0.1}  # [m]
MAX_CURVATURE = 5.0  # [1/m]
SOFT_CURVATURE = 0.8 * MAX_CURVATURE  # [1/m]
FREE_FORCE, MAX_FORCE = 2.0, 5.0  # [N]
MIN_MARGIN = 0.25  # [rad]
EPISODE_S = 10.0  # [s]
TIP_SEGMENT = 19
SPAWN_X, SPAWN_Y = (0.50, 0.65), (-0.15, 0.15)  # [m] stem spawn area (user, 2026-10-09), for the rest phase
# scripted motions
STEM_BASE = (0.65, 0.0)  # [m] stem base x, y for the scripted phases (far edge of the spawn area)
BEHIND = 0.20  # [m] tool tip this far behind the stem axis before a push (prongs reach 0.17 m beyond the tool tip)
PUSH_HEIGHT, IMPACT_HEIGHT = 0.30, 0.25  # [m] tool-tip height of the slow push and of the impact
REACH_TOL = 2e-3  # [m] the scripted drive must end this close to its goal, without a reset on the way
PUSH_DEPTH = 0.04  # [m] slow push: tool tip this far beyond first contact of the slot bottom
RAMP_FORCE, RAMP_STEPS = 2.0, 150  # [N], policy steps: tip force ramp of the curvature phase
REWARD_TOL = 1e-4
APPROACH_TOL = 5e-4  # [m]


def main() -> None:
    """Run the phases and print a pass/fail summary."""
    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        sim, device, n = env.sim, env.device, env.num_envs
        robot, stem, sensor = env.scene["robot"], env.scene["stem"], env.scene["stem_contact"]
        command = env.command_manager.get_term("stem_target")
        action_term = env.action_manager.get_term("tool_tip")
        model = stem_model(STEM_MODEL)
        geometry = stem_params(STEM_MODEL)["geometry"]
        segment_length = geometry["length"] / geometry["num_segments"]
        radius = 0.5 * geometry["diameter"]
        lengths = torch.full((geometry["num_segments"],), segment_length, device=device)
        dense = [i * 1e-3 for i in range(round(geometry["length"] * 1e3) + 1)]  # arc lengths every 1 mm
        arm = robot.find_joints("fr3_joint[1-7]")[0]
        max_step, max_step_change = action_term.cfg.max_step, action_term.cfg.max_step_change

        # -- independent values, taken when the termination manager starts (before the automatic reset)
        snap: dict[str, torch.Tensor] = {}

        def to_base(points_w: torch.Tensor) -> torch.Tensor:
            m = points_w.shape[1]
            root_pos = robot.data.root_pos_w.torch.repeat_interleave(m, dim=0)
            root_quat = robot.data.root_quat_w.torch.repeat_interleave(m, dim=0)
            return subtract_frame_transforms(root_pos, root_quat, points_w.reshape(-1, 3))[0].reshape(n, m, 3)

        def snapshot() -> None:
            poses = model.segment_poses(stem)
            tip = to_base(point_pose(poses, TIP_SEGMENT, 0.5 * segment_length)[:, None, :3])[:, 0]
            error = tip - command.command
            tool = action_term.tool_pose()[0]
            stem_b = to_base(stem_points(poses, dense, segment_length))
            data = sensor.data
            force = data.normal_force_matrix_w_history.torch + data.friction_force_matrix_w_history.torch
            q = robot.data.joint_pos.torch[:, arm]
            limits = robot.data.joint_pos_limits.torch[:, arm]
            snap.update(
                d=error.norm(dim=-1),
                dz=error[:, 2],
                approach_d=(stem_b - tool[:, None]).norm(dim=-1).min(dim=-1).values,
                curvature=joint_curvature(poses, lengths),
                force=force.norm(dim=-1).flatten(start_dim=2).max(dim=-1).values.mean(dim=1),  # largest single contact
                margin=torch.minimum(q - limits[..., 0], limits[..., 1] - q).min(dim=-1).values,
                tool=tool,
            )
            # the height term has weight 0 (the reward manager skips it): evaluated here, at the same state
            height_cfg = env.reward_manager.get_term_cfg("height")
            snap["height_value"] = height_cfg.func(env, **height_cfg.params)

        recorded: dict[str, torch.Tensor] = {}

        def wrap(manager, name: str, first: bool = False) -> None:
            term_cfg = manager.get_term_cfg(name)
            func = term_cfg.func

            def recording(env_, **params):
                if first:
                    snapshot()
                value = func(env_, **params)
                recorded[name] = value.clone()
                return value

            term_cfg.func = recording

        for i, name in enumerate(env.termination_manager.active_terms):
            wrap(env.termination_manager, name, first=i == 0)
        for name in env.reward_manager.active_terms:
            if name != "height":
                wrap(env.reward_manager, name)

        # -- checks of every step against the formulas
        errors = {key: 0.0 for key in ("rewards", "approach", "terminations", "height_error")}
        fired_wrongly: list[str] = []

        def tanh_reward(x: torch.Tensor, std: float) -> torch.Tensor:
            return 1.0 - torch.tanh(x / std)

        def check_step(done: torch.Tensor, contact_params: dict, limit_params: dict) -> None:
            s = snap
            expected = {
                "distance_coarse": tanh_reward(s["d"], STD["distance_coarse"]),
                "distance_fine": tanh_reward(s["d"], STD["distance_fine"]),
                "curvature": (s["curvature"] - SOFT_CURVATURE).clamp(min=0.0).square().sum(dim=-1),
                "contact_force": (s["force"] - contact_params["threshold"]).clamp(min=0.0).square(),
            }
            for name, value in expected.items():
                errors["rewards"] = max(errors["rewards"], float((recorded[name] - value).abs().max()))
            height = tanh_reward(s["dz"].abs(), STD["height"])
            errors["rewards"] = max(errors["rewards"], float((s["height_value"] - height).abs().max()))
            approach_d = STD["approach"] * torch.atanh((1.0 - recorded["approach"]).clamp(max=1.0 - 1e-7))
            errors["approach"] = max(errors["approach"], float((approach_d - s["approach_d"]).abs().max()))
            limits = {
                "curvature_limit": s["curvature"].max(dim=-1).values > MAX_CURVATURE,
                "contact_force_limit": s["force"] > limit_params["max_force"],
                "joint_margin": s["margin"] < MIN_MARGIN,
            }
            early = torch.zeros(n, dtype=torch.bool, device=device)
            for name, value in limits.items():
                early |= recorded[name]
                if not torch.equal(recorded[name], value):
                    fired_wrongly.append(f"{name} at step {env.common_step_counter}")
            errors["terminations"] = max(
                errors["terminations"], float((recorded["terminated"] - early.float()).abs().max())
            )
            kept = ~done  # the command's metric is computed after the reset: compare where no reset happened
            if kept.any():
                tip = to_base(point_pose(model.segment_poses(stem), TIP_SEGMENT, 0.5 * segment_length)[:, None, :3])
                dz = (tip[:, 0] - command.command)[:, 2]
                err = (command.metrics["height_error"] - dz)[kept].abs().max()
                errors["height_error"] = max(errors["height_error"], float(err))

        step_time = env.step_dt * args_cli.slow_motion
        clock = {"t": time.perf_counter()}
        contact_cfg = env.reward_manager.get_term_cfg("contact_force")
        limit_cfg = env.termination_manager.get_term_cfg("contact_force_limit")

        def step(action: torch.Tensor) -> torch.Tensor:
            """One policy step with all checks; returns the done flags (terminated or truncated)."""
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            check_step(done, contact_cfg.params, limit_cfg.params)
            if done.any() and not state["expect_reset"]:
                unexpected_resets.append(f"{state['phase']} at step {env.common_step_counter}")
            if sim.is_rendering:
                time.sleep(max(0.0, step_time - (time.perf_counter() - clock["t"])))
                clock["t"] = time.perf_counter()
            return done

        def zeros() -> torch.Tensor:
            return torch.zeros(n, 6, device=device)

        def toward(goal: torch.Tensor, speed: float = 1.0) -> torch.Tensor:
            """Action that moves the tool tip toward `goal` (n, 3), robot base frame, at most `speed` x the cap, and
            brakes in time for the step-change limit (stopping distance s^2 / (2 ds) at step s)."""
            offset = goal - action_term.tool_pose()[0]
            distance = offset.norm(dim=-1, keepdim=True)
            size = torch.sqrt(2.0 * max_step_change * distance).clamp(max=speed * max_step)
            action = zeros()
            action[:, :3] = offset / distance.clamp(min=1e-9) * size / max_step
            return action

        def place_stem(fixed: bool) -> None:
            """Stem base at STEM_BASE for the scripted phases, or the spawn area of the training."""
            params = env.event_manager.get_term_cfg("spawn_stem").params
            if fixed:
                params.update(x_range=(STEM_BASE[0],) * 2, y_range=(STEM_BASE[1],) * 2)
            else:
                params.update(x_range=SPAWN_X, y_range=SPAWN_Y)
            env.reset()

        def drive_to(goal: torch.Tensor, max_steps: int = 400) -> tuple[bool, str]:
            """Forward first, then down (straight down from the start pose reaches the joint margin after ~9 cm).
            Returns whether every env reached the goal without a reset, and a summary."""
            reset = False
            waypoint = goal.clone()
            waypoint[:, 2] = action_term.tool_pose()[0][:, 2]
            for target in (waypoint, goal):
                for _ in range(max_steps):
                    if (action_term.tool_pose()[0] - target).norm(dim=-1).max() < 1e-3:
                        break
                    reset |= bool(step(toward(target)).any())
            for _ in range(15):  # settle
                reset |= bool(step(toward(goal)).any())
            distance = float((action_term.tool_pose()[0] - goal).norm(dim=-1).max())
            return distance < REACH_TOL and not reset, f"drive to the goal: {distance * 1e3:.2f} mm off, reset: {reset}"

        def goal_behind(height: float) -> torch.Tensor:
            return torch.tensor([STEM_BASE[0] - BEHIND, STEM_BASE[1], height], device=device).repeat(n, 1)

        def f3(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        results: dict[str, tuple[bool, str]] = {}
        # a reset is expected only where a phase makes a term fire; elsewhere it fails the check
        state = {"phase": "", "expect_reset": False}
        unexpected_resets: list[str] = []
        # only the rest phase runs on the decided episode length; the scripted motions take longer than 10 s
        env.cfg.episode_length_s = 1e4

        for phase in args_cli.phases:
            state.update(phase=phase, expect_reset=phase in ("rest", "impact", "curvature", "joint_margin"))
            if phase == "rest":
                env.cfg.episode_length_s = EPISODE_S
                place_stem(fixed=False)
                steps = math.ceil(EPISODE_S / env.step_dt - 1e-9)
                fired_at, others, penalties, force, rate = None, False, 0.0, 0.0, 0.0
                for k in range(1, steps + 2):
                    step(zeros())
                    others |= bool(recorded["terminated"].any())
                    penalties = max(
                        penalties, float(recorded["curvature"].max()), float(recorded["contact_force"].max())
                    )
                    force = max(force, float(snap["force"].max()))
                    rate = max(rate, float(recorded["action_rate"].max()))
                    if fired_at is None and bool(recorded["time_out"].any()):
                        fired_at = (k, bool(recorded["time_out"].all()))
                        break
                env.cfg.episode_length_s = 1e4
                ok = fired_at == (steps, True) and not others and penalties == 0.0 and force == 0.0 and rate == 0.0
                results["rest / time out"] = (
                    ok,
                    f"time out at step {fired_at} (expected {steps}, all envs); other terminations: {others}; "
                    f"largest curvature / contact penalty {penalties:.2e}, contact force {force:.2e} N, "
                    f"action rate {rate:.2e}",
                )

            elif phase == "action_rate":
                env.reset()
                sequence = [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.0]
                previous, rate_error = 0.0, 0.0
                for a in sequence:
                    action = zeros()
                    action[:, 1] = a
                    step(action)
                    rate_error = max(rate_error, float((recorded["action_rate"] - (a - previous) ** 2).abs().max()))
                    previous = a
                results["action rate"] = (
                    rate_error < 1e-6,
                    f"largest difference to sum((a_t - a_t-1)^2): {rate_error:.2e}",
                )

            elif phase == "push":
                place_stem(fixed=True)
                step(zeros())
                approach_start = recorded["approach"].clone()
                reached, reach_info = drive_to(goal_behind(PUSH_HEIGHT))
                approach_behind = recorded["approach"].clone()
                force_before = float(snap["force"].max())
                contact_x = STEM_BASE[0] - radius  # slot bottom (tool tip) touches the stem surface
                goal = goal_behind(PUSH_HEIGHT)
                goal[:, 0] = contact_x - 0.01  # at full speed until 1 cm before contact (the prongs pass the stem)
                for _ in range(200):
                    if (action_term.tool_pose()[0] - goal).norm(dim=-1).max() < 1e-3:
                        break
                    step(toward(goal))
                goal[:, 0] = contact_x + PUSH_DEPTH
                max_force, max_penalty, max_curv = 0.0, 0.0, 0.0
                for _ in range(400):
                    step(toward(goal, speed=0.5))
                    max_force = max(max_force, float(snap["force"].max()))
                    max_penalty = max(max_penalty, float(recorded["contact_force"].max()))
                    max_curv = max(max_curv, float(snap["curvature"].max()))
                    if (snap["tool"][:, 0] >= goal[:, 0] - 1e-3).all():
                        break
                approach_contact = float(recorded["approach"].min())
                min_force_end = float(snap["force"].min())
                results["push: approach"] = (
                    reached and bool((approach_behind > approach_start + 0.01).all()) and approach_contact > 0.9,
                    f"{reach_info}; approach at the start {float(approach_start.mean()):.3f}, behind the stem "
                    f"{float(approach_behind.mean()):.3f}, in contact (min) {approach_contact:.3f}",
                )
                results["push: contact"] = (
                    force_before == 0.0 and min_force_end > 0.05 and max_force < FREE_FORCE and max_penalty == 0.0,
                    f"force before the push {force_before:.2e} N; during the push up to {max_force:.3f} N, at the end "
                    f"at least {min_force_end:.3f} N; largest penalty {max_penalty:.2e}; largest curvature "
                    f"{max_curv:.2f} 1/m",
                )
                # mechanism with lowered thresholds (not the decided values), restored afterwards
                contact_cfg.params["threshold"] = 0.0
                step(toward(goal, speed=0.5))
                square_error = float((recorded["contact_force"] - snap["force"].square()).abs().max())
                penalty_zero = float(recorded["contact_force"].min())
                contact_cfg.params["threshold"] = FREE_FORCE
                limit_cfg.params["max_force"] = 0.5 * float(snap["force"].min())
                lowered = limit_cfg.params["max_force"]
                state["expect_reset"] = True
                step(toward(goal, speed=0.5))
                fired = bool(recorded["contact_force_limit"].all())
                limit_cfg.params["max_force"] = MAX_FORCE
                results["push: contact mechanism (lowered thresholds)"] = (
                    square_error < REWARD_TOL and penalty_zero > 0.0 and fired,
                    f"threshold 0 N: penalty = F^2 within {square_error:.1e} (smallest {penalty_zero:.3f}); limit "
                    f"{lowered:.3f} N: fired in every env: {fired}",
                )

            elif phase == "impact":
                place_stem(fixed=True)
                reached, reach_info = drive_to(goal_behind(IMPACT_HEIGHT))
                goal = goal_behind(IMPACT_HEIGHT)
                goal[:, 0] = STEM_BASE[0] + 0.05
                max_force, fired, margin_fired = 0.0, False, False
                for _ in range(150):
                    step(toward(goal))
                    max_force = max(max_force, float(snap["force"].max()))
                    fired |= bool(recorded["contact_force_limit"].any())
                    margin_fired |= bool(recorded["joint_margin"].any())
                    if fired or margin_fired:
                        break
                results["impact (informational)"] = (
                    reached and max_force > 0.0,
                    f"{reach_info}; largest mean force of a policy step {max_force:.2f} N (limit {MAX_FORCE} N), "
                    f"contact limit "
                    f"fired: {fired}, joint margin fired: {margin_fired}",
                )

            elif phase == "curvature":
                place_stem(fixed=True)
                set_force = model.segment_force_setter(stem)
                force = torch.zeros(n, 3, device=device)
                first_penalty, fired, early = None, None, False
                for k in range(1, RAMP_STEPS + 1):
                    force[:, 0] = RAMP_FORCE * k / RAMP_STEPS
                    set_force(TIP_SEGMENT, force)
                    step(zeros())
                    curv = float(snap["curvature"].max())
                    if first_penalty is None and bool((recorded["curvature"] > 0).any()):
                        first_penalty = (float(force[0, 0]), curv)
                    early |= bool((recorded["contact_force_limit"] | recorded["joint_margin"]).any())
                    if recorded["curvature_limit"].any():
                        fired = (float(force[0, 0]), curv, bool(recorded["curvature_limit"].all()))
                        break
                set_force(TIP_SEGMENT, None)
                ok = (
                    first_penalty is not None
                    and SOFT_CURVATURE < first_penalty[1] <= MAX_CURVATURE
                    and fired is not None
                    and fired[1] > MAX_CURVATURE
                    and fired[2]
                    and not early
                )
                results["curvature"] = (
                    ok,
                    f"penalty first > 0 at tip force {f3(first_penalty and first_penalty[0])} N, curvature "
                    f"{f3(first_penalty and first_penalty[1])} 1/m; limit fired at {f3(fired and fired[0])} N, "
                    f"{f3(fired and fired[1])} 1/m, in every env: {fired and fired[2]}; other terminations: {early}",
                )

            elif phase == "joint_margin":
                place_stem(fixed=True)
                fired, margins = None, []
                for k in range(1, 200):
                    action = zeros()
                    action[:, 2] = -1.0
                    step(action)
                    margins.append(float(snap["margin"].min()))
                    if recorded["joint_margin"].any():
                        fired = (k, float(snap["tool"][:, 2].mean()), bool(recorded["joint_margin"].all()))
                        break
                ok = fired is not None and fired[2] and margins[-1] < MIN_MARGIN and min(margins[:-1]) >= MIN_MARGIN
                results["joint margin"] = (
                    ok,
                    f"fired at step {fired and fired[0]}, tool tip z {f3(fired and fired[1])} m, in every env: "
                    f"{fired and fired[2]}; margin at the start {margins[0]:.3f} rad, at firing {margins[-1]:.3f} rad",
                )

        results["no unexpected reset"] = (
            not unexpected_resets,
            f"resets outside the intended firings: {unexpected_resets[:5] or 'none'}",
        )
        results["every step: terms = formulas"] = (
            errors["rewards"] < REWARD_TOL
            and errors["approach"] < APPROACH_TOL
            and errors["terminations"] == 0.0
            and errors["height_error"] < 1e-5
            and not fired_wrongly,
            f"rewards {errors['rewards']:.1e}, approach distance {errors['approach'] * 1e3:.3f} mm, terminated "
            f"{errors['terminations']:.0f}, height_error {errors['height_error'] * 1e3:.4f} mm; terminations not "
            f"matching their limit: {fired_wrongly[:5] or 'none'}",
        )
        print(f"\n=== check_push_terms ({n} envs, phases {' '.join(args_cli.phases)}) ===")
        for name, (ok, info) in results.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        print("=== all passed ===" if all(ok for ok, _ in results.values()) else "=== FAILED ===")
        env.close()


if __name__ == "__main__":
    main()
