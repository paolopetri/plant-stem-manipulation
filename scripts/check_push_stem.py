"""Push check for stage 1 (`StemManip-Push-Position-FR3-v0`): the fork pushes the stem with the task's action.

The slot base is driven by the task's action term (Franka's Cartesian impedance law with its speed cap, step-change
limit and setpoint clamp, `mdp.SlotBaseImpedanceAction`), scripted with `rollout.toward`. The stem base stands at
(0.65, 0) m. The contact force is the force of the robot on the stem from the task's stem contact sensor
(`stem_contact`, normal + friction), summed over all stem segments and robot bodies, at every physics step: the
spring bound of the clamp holds for the total force, not for a single contact.

Phases:
- normal push: the fork behind the stem at 0.30 m height, at full speed until 1 cm before contact, then 10 cm into
  the slot at 5 cm/s (as `check_push_terms.py`, deeper: the force must cover the normal push forces), hold, pull
  back, wait;
- hard push: the fork at 0.05 m height (the stem is stiff there: 4 N bend it by about 1 mm), at full speed until 1 cm
  before contact, then a full action 10 cm beyond the stem, which the stem blocks.
Checks (#77; decided values fixed here, not read from the cfg):
- deflects: the stem tip moves more than 1 cm during the normal push (as `check_stem.py`);
- springs back: the stem tip is back within 1 mm of its rest position 3 s after pulling back (as
  `check_contact.py`);
- clamp inactive at normal push forces: during the normal push the clamp never acts (position or orientation), and
  the largest contact force reaches at least 1.4 N (it covers the normal push forces measured, 0.8-1.4 N;
  2026-10-08);
- hard push: the contact force stays <= 4 N (clamp 4 mm x K_p 1000; user, 2026-10-08) at every physics step from
  first contact on, in two rows: until blocked (the setpoint runs ahead of the tool; the clamp has not engaged yet),
  and blocked (the clamp holds the position setpoint at every physics step of the last 20 policy steps). The
  contact-force limit (a damage limit, 5 N) is lifted for this phase so that the blocked state is reached;
- no chatter: in the blocked window the contact force varies by at most 0.2 N (5 % of the bound) over its physics
  steps (check design, not a decided value);
- no reset in any phase (the task's terminations stay active, except the lifted contact-force limit).

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/check_push_stem.py
    uv run --extra isaacsim python scripts/check_push_stem.py --num_envs 1 --viz kit --slow_motion 3
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Push check with the task's action for the stage-1 push task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs.")
parser.add_argument(
    "--slow_motion", type=float, default=1.0, help="With a viewer: play this many times slower than real time."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import time

import gymnasium as gym
import torch

import stem_manip.tasks  # noqa: F401  (registers the task)
from stem_manip.assets.stem import stem_model, stem_params
from stem_manip.tasks.push_position.env_cfg import STEM_MODEL, StemPushPositionEnvCfg
from stem_manip.utils import check_report, rollout
from stem_manip.utils.stem_geometry import point_pose

TASK = "StemManip-Push-Position-FR3-v0"
# decided values (#77; user, 2026-10-08)
MAX_FORCE = 4.0  # [N] spring bound: setpoint clamp 4 mm x K_p 1000
CLAMP_OFFSET = 4e-3  # [m] setpoint clamp (2026-10-08)
NORMAL_FORCE = 1.4  # [N] upper end of the measured normal push forces (0.8-1.4 N)
MIN_TIP_DEFLECTION = 0.01  # [m] as check_stem.py
REST_TOL = 1e-3  # [m] as check_contact.py
REST_TIME = 3.0  # [s] wait after pulling back, as check_contact.py
# check design (not decided values)
CHATTER_TOL = 0.05 * MAX_FORCE  # [N] largest spread of the contact force over the blocked window
BLOCKED_STEPS = 20  # policy steps at the end of the hard push
HARD_STEPS = 120  # policy steps of the hard push from 1 cm before contact
CONTACT_THRESHOLD = 1e-3  # [N] first contact (as check_contact.py)
TIP_SEGMENT = 19
# scripted motions (as check_push_terms.py)
STEM_BASE = (0.65, 0.0)  # [m] stem base x, y (far edge of the spawn area)
BEHIND = 0.20  # [m] slot base this far behind the stem axis before a push (prongs reach 0.17 m beyond the slot base)
PUSH_HEIGHT, HARD_HEIGHT = 0.30, 0.05  # [m] slot-base height of the normal and the hard push
PUSH_DEPTH, HARD_DEPTH = 0.10, 0.10  # [m] slot base this far beyond first contact of the slot bottom (goal)
HOLD_STEPS = 45  # policy steps of the hold after the normal push
REACH_TOL = 2e-3  # [m] the scripted drive must end this close to its goal, without a reset on the way


def main() -> None:
    """Run the phases and print a pass/fail summary."""
    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.episode_length_s = 1e4  # the scripted phases run without a time out
    env_cfg.events.spawn_stem.params.update(x_range=(STEM_BASE[0],) * 2, y_range=(STEM_BASE[1],) * 2)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        sim, device, n = env.sim, env.device, env.num_envs
        stem, sensor = env.scene["stem"], env.scene["stem_contact"]
        term = env.action_manager.get_term("slot_base")
        model = stem_model(STEM_MODEL)
        geometry = stem_params(STEM_MODEL)["geometry"]
        segment_length = geometry["length"] / geometry["num_segments"]
        radius = 0.5 * geometry["diameter"]
        max_step, max_step_change = term.cfg.max_step, term.cfg.max_step_change

        resets: list[str] = []
        step_time = env.step_dt * args_cli.slow_motion
        clock = {"t": time.perf_counter()}
        # per physics step of the latest policy step: total contact force [N] (n, decimation), setpoint-tool distance
        forces = torch.zeros(n, env.cfg.decimation, device=device)
        stretches: list[torch.Tensor] = []
        apply_actions = term.apply_actions

        def apply_and_measure() -> None:
            pos = term.slot_base_pose()[0].clone()  # the pose the law sees in this physics step
            apply_actions()
            stretches.append((term.command_pose()[0] - pos).norm(dim=-1))

        term.apply_actions = apply_and_measure

        def step(action: torch.Tensor, phase: str) -> torch.Tensor:
            """One policy step; returns the setpoint-tool distance at each of its physics steps (n, decimation)."""
            stretches.clear()
            _, _, terminated, truncated, _ = env.step(action)
            if (terminated | truncated).any():
                manager = env.termination_manager
                fired = [name for name in manager.active_terms if manager.get_term(name).any()]
                resets.append(f"{phase} at step {env.common_step_counter} ({', '.join(fired)})")
            data = sensor.data
            total = (data.normal_force_matrix_w_history.torch + data.friction_force_matrix_w_history.torch).sum(
                dim=(2, 3)
            )  # (n, history = decimation, 3), newest first
            forces[:] = total.norm(dim=-1).flip(dims=(1,))
            if sim.is_rendering:
                time.sleep(max(0.0, step_time - (time.perf_counter() - clock["t"])))
                clock["t"] = time.perf_counter()
            return torch.stack(stretches, dim=1)

        def tip() -> torch.Tensor:
            """Stem tip position (n, 3), world frame."""
            return point_pose(model.segment_poses(stem), TIP_SEGMENT, 0.5 * segment_length)[:, :3]

        def toward(goal: torch.Tensor, speed: float = 1.0) -> torch.Tensor:
            return rollout.toward(term.slot_base_pose()[0], goal, max_step, max_step_change, speed)

        def goal_at(x: float, height: float) -> torch.Tensor:
            return torch.tensor([x, STEM_BASE[1], height], device=device).repeat(n, 1)

        def drive_to(goal: torch.Tensor, phase: str, speed: float = 1.0, max_steps: int = 400) -> float:
            """Forward / back first, then down (as check_push_terms.py); returns the largest distance to the goal."""
            waypoint = goal.clone()
            waypoint[:, 2] = term.slot_base_pose()[0][:, 2]
            for target in (waypoint, goal):
                for _ in range(max_steps):
                    if (term.slot_base_pose()[0] - target).norm(dim=-1).max() < 1e-3:
                        break
                    step(toward(target, speed), phase)
            for _ in range(15):  # settle
                step(toward(goal, speed), phase)
            return float((term.slot_base_pose()[0] - goal).norm(dim=-1).max())

        contact_x = STEM_BASE[0] - radius  # slot bottom (slot base) touches the stem surface
        results: dict[str, tuple[bool, str]] = {}

        # -- normal push
        env.reset()
        rest_tip = tip().clone()
        behind = drive_to(goal_at(STEM_BASE[0] - BEHIND, PUSH_HEIGHT), "normal push")
        before = drive_to(goal_at(contact_x - 0.01, PUSH_HEIGHT), "normal push")
        goal = goal_at(contact_x + PUSH_DEPTH, PUSH_HEIGHT)
        clamp_acted, max_force, deflection = False, 0.0, torch.zeros(n, device=device)
        for _ in range(400):
            step(toward(goal, speed=0.5), "normal push")
            clamp_acted |= bool(term.clamp_held().any())
            max_force = max(max_force, float(forces.max()))
            deflection = torch.maximum(deflection, (tip() - rest_tip).norm(dim=-1))
            if (term.slot_base_pose()[0] - goal).norm(dim=-1).max() < 1e-3:
                break
        for _ in range(HOLD_STEPS):
            step(toward(goal, speed=0.5), "normal push")
            clamp_acted |= bool(term.clamp_held().any())
            max_force = max(max_force, float(forces.max()))
            deflection = torch.maximum(deflection, (tip() - rest_tip).norm(dim=-1))
        pushed = float((term.slot_base_pose()[0] - goal).norm(dim=-1).max())
        drive_to(goal_at(STEM_BASE[0] - BEHIND, PUSH_HEIGHT), "pull back", speed=0.5)
        for _ in range(round(REST_TIME / env.step_dt)):
            step(torch.zeros(n, 6, device=device), "rest")
        tip_rest = float((tip() - rest_tip).norm(dim=-1).max())
        reach = (
            f"drive behind {behind * 1e3:.2f} mm, before contact {before * 1e3:.2f} mm, push {pushed * 1e3:.2f} mm off"
        )
        results["normal push: deflects"] = (
            max(behind, before, pushed) < REACH_TOL and float(deflection.min()) > MIN_TIP_DEFLECTION,
            f"{reach}; tip moved up to {float(deflection.min()) * 1e3:.1f}-{float(deflection.max()) * 1e3:.1f} mm "
            f"(> {MIN_TIP_DEFLECTION * 1e3:.0f} mm)",
        )
        results["normal push: springs back"] = (
            tip_rest < REST_TOL,
            f"stem tip {tip_rest * 1e3:.2f} mm from rest {REST_TIME:.0f} s after pulling back (tol "
            f"{REST_TOL * 1e3:.0f} mm)",
        )
        results["normal push: clamp inactive"] = (
            not clamp_acted and max_force >= NORMAL_FORCE,
            f"clamp acted: {clamp_acted}; largest contact force {max_force:.3f} N (normal push >= {NORMAL_FORCE} N)",
        )

        # -- hard push: blocked by the stiff lower stem. The contact-force limit (a damage limit, not part of this
        # check) is lifted for this phase, so that the blocked state is reached; its reading is reported.
        env.reset()
        behind = drive_to(goal_at(STEM_BASE[0] - BEHIND, HARD_HEIGHT), "hard push")
        before = drive_to(goal_at(contact_x - 0.01, HARD_HEIGHT), "hard push")
        limit_cfg = env.termination_manager.get_term_cfg("contact_force_limit")
        damage_limit = limit_cfg.params["max_force"]
        limit_cfg.params["max_force"] = float("inf")
        goal = goal_at(contact_x + HARD_DEPTH, HARD_HEIGHT)
        touched, transient_force, step_mean = False, 0.0, 0.0
        window_forces, window_stretch = [], []
        for i in range(HARD_STEPS):
            stretch = step(toward(goal), "hard push")
            touched |= bool((forces > CONTACT_THRESHOLD).any())
            step_mean = max(step_mean, float(forces.mean(dim=1).max()))
            if i >= HARD_STEPS - BLOCKED_STEPS:
                window_forces.append(forces.clone())
                window_stretch.append(stretch.clone())
            elif touched:
                transient_force = max(transient_force, float(forces.max()))
        limit_cfg.params["max_force"] = damage_limit
        term.apply_actions = apply_actions
        window_forces = torch.cat(window_forces, dim=1)  # (n, BLOCKED_STEPS x decimation)
        window_stretch = torch.cat(window_stretch, dim=1)
        held = bool(((window_stretch - CLAMP_OFFSET).abs() < 1e-6).all())  # held: setpoint at the clamp's bound
        spread = float((window_forces.max(dim=1).values - window_forces.min(dim=1).values).max())
        window_means = window_forces.unflatten(1, (BLOCKED_STEPS, -1)).mean(dim=-1)  # mean over each policy step
        tool_x = term.slot_base_pose()[0][:, 0]
        results["hard push: force <= 4 N until blocked"] = (
            max(behind, before) < REACH_TOL and touched and transient_force <= MAX_FORCE,
            f"drive behind {behind * 1e3:.2f} mm, before contact {before * 1e3:.2f} mm off; largest contact force "
            f"from first contact to the blocked window {transient_force:.3f} N (bound {MAX_FORCE} N); largest mean "
            f"over a policy step over the whole push {step_mean:.3f} N (damage limit {damage_limit} N, lifted here)",
        )
        results["hard push: blocked, force <= 4 N"] = (
            held and float(window_forces.max()) <= MAX_FORCE,
            f"clamp held at every physics step of the last {BLOCKED_STEPS} policy steps: {held} (setpoint-tool "
            f"distance {float(window_stretch.min()) * 1e3:.4f}-{float(window_stretch.max()) * 1e3:.4f} mm); contact "
            f"force {float(window_forces.min()):.3f}-{float(window_forces.max()):.3f} N (mean over a policy step "
            f"{float(window_means.min()):.3f}-{float(window_means.max()):.3f} N; bound {MAX_FORCE} N); slot "
            f"base {float((tool_x - contact_x).min()) * 1e3:.1f}-{float((tool_x - contact_x).max()) * 1e3:.1f} mm "
            f"beyond first contact",
        )
        results["hard push: no chatter"] = (
            held and spread <= CHATTER_TOL,
            f"blocked: largest spread of the contact force in one env {spread:.3f} N (tol {CHATTER_TOL:.2f} N)",
        )
        results["no reset"] = (not resets, f"resets: {resets[:5] or 'none'}")
        print(f"\n=== check_push_stem ({n} envs) ===")
        env.close()
        check_report.finish(results)


if __name__ == "__main__":
    main()
