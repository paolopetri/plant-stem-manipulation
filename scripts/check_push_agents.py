"""Env check for stage 1 (`StemManip-Push-Position-FR3-v0`): zero and random agent, shapes and value ranges.

Runs the task as Isaac Lab's `isaaclab zero_agent` / `random_agent` do (zero actions; uniform random actions in
[-1, 1], seeded) for `--episodes` episodes and checks every policy step. The expected values are the decided ones,
fixed here, not read from the cfg:
- shapes: observation (num_envs, 57), action (num_envs, 6); terms tool_tip_pos 3, tool_tip_rot 6, applied_step 6,
  target_offset 6, stem_base 3, stem_points 30 (5 points, previous and current policy step), target_pos 3;
- finite: observations, rewards, terminated / truncated flags;
- tool_tip_rot: both columns unit length and orthogonal (1e-4);
- after every reset: tool tip within 5 mm of the start pose (0.40, 0, 0.50) m and within 1 deg (rotation angle) of
  the orientation at the first reset (tolerances of `check_push_env.py`);
- target_offset: position part and rotation part each of norm <= 1 (target clamp 4 mm / 3 deg, 2026-10-08);
- applied_step: translation and rotation part each of norm <= 1 (speed caps), except at steps where the clamp holds
  the target (target_offset part >= 1 - 1e-3: there the clamp drags the target along, see `mdp.applied_step`);
- stem_base: in the spawn area x 0.50-0.65 m, y +-0.15 m (1e-6 m), z 0 (1 mm), as in `check_push_obs.py`;
  stem base constant within an episode (1e-6 m, user 2026-10-09: the stem stays bit-identical in the world, but the
  robot root pose PhysX returns after the first physics step differs from the written one by ~1e-8 in the
  quaternion and up to 1 float32 ulp in position, which moves the base in the robot base frame by up to 5e-8 m),
  target_pos exactly constant;
- stem_points: each point no farther from the base than its arc length 0.08 ... 0.40 m (+1 mm: inextensible chain);
- rewards (weighted, per step): distance / approach terms in [0, weight], penalties <= 0; height 0 only confirms its
  weight 0 (Isaac Lab skips a term of weight 0 without calling it, so its function is not tested here);
- every reset comes with a done flag (episode length counts up by one, or restarts after terminated / truncated).
Zero agent in addition: only the time out fires, exactly at step 469 (15 s / 32 ms) of every episode in every env;
the tool tip stays within 2 mm of its reset position (lag criterion of the action sweep); curvature, contact and
action-rate penalties 0. Random agent: the terminations that fire are reported (counts per term), not checked.

Every loop has a fixed number of steps and a progress line is printed every 100 steps. The check fails when it takes
longer than `--max_minutes`; the limit is checked between policy steps, so a hang inside `env.step` (or at startup)
is not caught: run it under an outer `timeout` when it runs unattended. Rows of an agent cut short by the limit
print NOT RUN unless they had already failed.

Usage (from the repo root; headless unless a visualizer is requested):
    uv run --extra isaacsim python scripts/check_push_agents.py
    uv run --extra isaacsim python scripts/check_push_agents.py --agents random --num_envs 16
    uv run --extra isaacsim python scripts/check_push_agents.py --agents random --num_envs 4 --viz kit --slow_motion 2
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

AGENTS = ("zero", "random")
parser = argparse.ArgumentParser(description="Zero / random agent check for the stage-1 push task.")
parser.add_argument("--agents", nargs="+", default=list(AGENTS), choices=AGENTS, help="Agents to run, in order.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs.")
parser.add_argument("--episodes", type=int, default=2, help="Episodes (of 469 steps) per agent.")
parser.add_argument("--max_minutes", type=float, default=15.0, help="Wall-clock limit of the whole check [min].")
parser.add_argument("--seed", type=int, default=42, help="Seed of the random actions.")
parser.add_argument(
    "--slow_motion", type=float, default=1.0, help="With a viewer: play this many times slower than real time."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import math
import time

import gymnasium as gym
import torch

import stem_manip.tasks  # noqa: F401  (registers the task)
from stem_manip.tasks.push_position.env_cfg import StemPushPositionEnvCfg

TASK = "StemManip-Push-Position-FR3-v0"
# decided values
OBS_DIM, ACTION_DIM = 57, 6
TERM_DIMS = {
    "tool_tip_pos": 3,
    "tool_tip_rot": 6,
    "applied_step": 6,
    "target_offset": 6,
    "stem_base": 3,
    "stem_points": 30,
    "target_pos": 3,
}
ARC_LENGTHS = (0.08, 0.16, 0.24, 0.32, 0.40)  # [m] observed stem points (2026-10-08)
EPISODE_STEPS = 469  # 15 s / (16 x 2 ms), rounded up (user, 2026-10-09)
START_POS = (0.40, 0.0, 0.50)  # [m] tool tip of the start pose (user, 2026-10-09)
START_TOL, START_ANGLE_TOL = 5e-3, 1.0  # [m], [deg] as in check_push_env.py
HOLD_TOL = 2e-3  # [m] zero agent: tool tip vs its reset position (lag criterion of the action sweep)
SPAWN_X, SPAWN_Y = (0.50, 0.65), (-0.15, 0.15)  # [m] stem spawn area (user, 2026-10-09)
AREA_TOL, GROUND_TOL = 1e-6, 1e-3  # [m] as in check_push_obs.py
ARC_TOL = 1e-3  # [m]
UNIT_TOL, NORM_TOL, CLAMP_ACTIVE = 1e-4, 1e-3, 1.0 - 1e-3
POSITIVE_REWARDS = ("distance_coarse", "distance_fine", "approach")  # tanh terms in [0, weight]
PENALTIES = ("curvature", "contact_force", "action_rate", "terminated")
ZERO_PENALTIES = ("curvature", "contact_force", "action_rate")  # zero agent: exactly 0
PROGRESS_EVERY = 100  # [policy steps]


def main() -> None:
    """Run the agents and print a pass/fail summary."""
    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    t_start = time.perf_counter()
    deadline = t_start + 60.0 * args_cli.max_minutes
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        sim, device, n = env.sim, env.device, env.num_envs
        weights = {name: env.reward_manager.get_term_cfg(name).weight for name in env.reward_manager.active_terms}
        reward_names = env.reward_manager.active_terms
        term_names = env.termination_manager.active_terms
        obs_names = env.observation_manager.active_terms["policy"]
        obs_dims = dict(zip(obs_names, env.observation_manager.group_obs_term_dim["policy"], strict=True))
        slices, start = {}, 0
        for name in obs_names:
            size = math.prod(obs_dims[name])
            slices[name] = slice(start, start + size)
            start += size
        results: dict[str, tuple[bool, str]] = {}
        results["shapes"] = (
            {k: math.prod(v) for k, v in obs_dims.items()} == TERM_DIMS
            and tuple(env.observation_space["policy"].shape) == (n, OBS_DIM)
            and tuple(env.action_space.shape) == (n, ACTION_DIM),
            f"obs {tuple(env.observation_space['policy'].shape)}, action {tuple(env.action_space.shape)}, terms "
            + ", ".join(f"{k} {math.prod(v)}" for k, v in obs_dims.items()),
        )
        arc = torch.tensor(ARC_LENGTHS, device=device)
        start_pos = torch.tensor(START_POS, device=device)
        step_time = env.step_dt * args_cli.slow_motion
        generator = torch.Generator(device=device).manual_seed(args_cli.seed)
        timed_out = False
        results_not_run: set[str] = set()  # rows of an agent cut short by the wall-clock limit

        def part_norms(x: torch.Tensor) -> torch.Tensor:
            """Norms of the translation and rotation parts (n, 2) of a 6-D term."""
            return torch.stack((x[:, :3].norm(dim=-1), x[:, 3:].norm(dim=-1)), dim=-1)

        for agent in args_cli.agents:
            obs = env.reset()[0]["policy"]
            rot_ref = obs[:, slices["tool_tip_rot"]].clone()  # orientation at the first reset
            fails: dict[str, list[str]] = {}
            stats = {"step_out": 0.0, "offset": 0.0, "unit": 0.0, "arc": -1.0, "hold": 0.0, "start": 0.0, "angle": 0.0}
            fired = {name: 0 for name in term_names}
            returns = torch.zeros(n, device=device)
            episode_returns: list[float] = []
            episode_len = torch.zeros(n, dtype=torch.long, device=device)

            # this agent's state bound as defaults (the functions are redefined per agent)
            def fail(check: str, info: str, fails: dict = fails) -> None:
                fails.setdefault(check, []).append(info)

            def check_reset(
                obs: torch.Tensor, ids: torch.Tensor, step: int, rot_ref: torch.Tensor = rot_ref, stats: dict = stats
            ) -> None:
                """Start pose and stem base of the envs `ids` right after their reset."""
                if len(ids) == 0:
                    return
                o = obs[ids]
                err = float((o[:, slices["tool_tip_pos"]] - start_pos).norm(dim=-1).max())
                # angle between the orientations: trace(R_ref^T R) = 1 + 2 cos(angle); third column = x cross y
                rot, ref = o[:, slices["tool_tip_rot"]], rot_ref[ids]
                trace = sum(
                    (a * b).sum(-1)
                    for a, b in (
                        (rot[:, :3], ref[:, :3]),
                        (rot[:, 3:], ref[:, 3:]),
                        (torch.linalg.cross(rot[:, :3], rot[:, 3:]), torch.linalg.cross(ref[:, :3], ref[:, 3:])),
                    )
                )
                angle = float(torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1.0, 1.0))).max())
                stats["start"], stats["angle"] = max(stats["start"], err), max(stats["angle"], angle)
                if err > START_TOL or angle > START_ANGLE_TOL:
                    fail("start pose after reset", f"step {step}: {err * 1e3:.2f} mm, {angle:.2f} deg")
                base = o[:, slices["stem_base"]]
                x, y, z = base.unbind(-1)
                inside = (
                    (x >= SPAWN_X[0] - AREA_TOL)
                    & (x <= SPAWN_X[1] + AREA_TOL)
                    & (y >= SPAWN_Y[0] - AREA_TOL)
                    & (y <= SPAWN_Y[1] + AREA_TOL)
                    & (z.abs() <= GROUND_TOL)
                )
                if not inside.all():
                    fail("stem base in the spawn area", f"step {step}: {base[~inside].tolist()}")

            check_reset(obs, torch.arange(n, device=device), 0)
            reset_pos = obs[:, slices["tool_tip_pos"]].clone()
            prev_obs = obs.clone()
            steps = args_cli.episodes * EPISODE_STEPS + 2
            print(f"[INFO] {agent} agent: {steps} steps, {n} envs", flush=True)
            t_agent = time.perf_counter()
            steps_run = 0
            for step in range(1, steps + 1):
                if time.perf_counter() > deadline:
                    timed_out = True
                    fail("wall-clock limit", f"{args_cli.max_minutes} min exceeded at step {step}")
                    break
                steps_run = step
                frame_start = time.perf_counter()
                if agent == "zero":
                    action = torch.zeros(n, ACTION_DIM, device=device)
                else:
                    action = 2 * torch.rand(n, ACTION_DIM, device=device, generator=generator) - 1
                obs_dict, reward, terminated, truncated, _ = env.step(action)
                obs = obs_dict["policy"]
                done = terminated | truncated
                step_reward = env.reward_manager._step_reward  # weighted reward per term (before x dt)
                # termination flags of this step (before the reset)
                term_flags = {name: env.termination_manager.get_term(name) for name in term_names}

                # finite
                if not (
                    torch.isfinite(obs).all() and torch.isfinite(reward).all() and torch.isfinite(step_reward).all()
                ):
                    fail("finite", f"step {step}")
                if tuple(obs.shape) != (n, OBS_DIM):
                    fail("shapes", f"step {step}: obs {tuple(obs.shape)}")

                # ranges of every step
                rot = obs[:, slices["tool_tip_rot"]]
                unit = torch.stack(
                    (rot[:, :3].norm(dim=-1) - 1, rot[:, 3:].norm(dim=-1) - 1, (rot[:, :3] * rot[:, 3:]).sum(-1)),
                    dim=-1,
                )
                stats["unit"] = max(stats["unit"], float(unit.abs().max()))
                offset = part_norms(obs[:, slices["target_offset"]])
                applied = part_norms(obs[:, slices["applied_step"]])
                stats["offset"] = max(stats["offset"], float(offset.max()))
                free = offset < CLAMP_ACTIVE  # the clamp does not hold that part's target
                stats["step_out"] = max(stats["step_out"], float(applied[free].max()) if free.any() else 0.0)
                if (offset > 1 + NORM_TOL).any():
                    fail("target_offset <= 1", f"step {step}: {offset.max():.4f}")
                if (applied[free] > 1 + UNIT_TOL).any():
                    fail("applied_step <= 1 (clamp free)", f"step {step}: {applied[free].max():.4f}")
                base = obs[:, slices["stem_base"]]
                points = obs[:, slices["stem_points"]].reshape(n, 2, len(ARC_LENGTHS), 3)
                excess = (points - base[:, None, None]).norm(dim=-1) - arc
                stats["arc"] = max(stats["arc"], float(excess.max()))
                if (excess > ARC_TOL).any():
                    fail("stem points within their arc length", f"step {step}: {excess.max() * 1e3:.2f} mm")
                for name in POSITIVE_REWARDS:
                    value = step_reward[:, reward_names.index(name)]
                    if ((value < 0) | (value > weights[name] + UNIT_TOL)).any():
                        fail("reward ranges", f"step {step}: {name} {value.min():.4f}..{value.max():.4f}")
                for name in PENALTIES:
                    if (step_reward[:, reward_names.index(name)] > 0).any():
                        fail("reward ranges", f"step {step}: {name} > 0")
                if (step_reward[:, reward_names.index("height")] != 0).any():
                    fail("reward ranges", f"step {step}: height != 0")

                # episode bookkeeping: constant within an episode, reset only with a done flag
                kept = ~done
                for name, tol in (("stem_base", AREA_TOL), ("target_pos", 0.0)):
                    change = (obs[kept, slices[name]] - prev_obs[kept, slices[name]]).abs()
                    if kept.any() and change.max() > tol:
                        fail(f"{name} constant within an episode", f"step {step}: {change.max():.2e} m")
                episode_len = torch.where(done, torch.zeros_like(episode_len), episode_len + 1)
                if not torch.equal(env.episode_length_buf, episode_len):
                    fail("reset only with a done flag", f"step {step}")
                    episode_len = env.episode_length_buf.clone()
                returns += reward  # Isaac Lab's reward already includes weight x dt
                for name, flags in term_flags.items():
                    fired[name] += int(flags.sum())
                ids = done.nonzero().flatten()
                if len(ids):
                    episode_returns += returns[ids].tolist()
                    returns[ids] = 0.0

                # zero agent
                if agent == "zero":
                    early = torch.zeros(n, dtype=torch.bool, device=device)
                    for name, flags in term_flags.items():
                        if name != "time_out":
                            early |= flags
                    expected_out = step % EPISODE_STEPS == 0
                    if early.any() or not bool((term_flags["time_out"] == expected_out).all()):
                        fail("zero agent: only the time out, at step 469", f"step {step}")
                    hold = (obs[kept, slices["tool_tip_pos"]] - reset_pos[kept]).norm(dim=-1)
                    stats["hold"] = max(stats["hold"], float(hold.max()) if kept.any() else 0.0)
                    if (hold > HOLD_TOL).any():
                        fail("zero agent: tool tip holds", f"step {step}: {hold.max() * 1e3:.2f} mm")
                    for name in ZERO_PENALTIES:
                        if (step_reward[:, reward_names.index(name)] != 0).any():
                            fail("zero agent: penalties 0", f"step {step}: {name}")

                check_reset(obs, ids, step)
                reset_pos[ids] = obs[ids, slices["tool_tip_pos"]]
                prev_obs = obs.clone()
                if step % PROGRESS_EVERY == 0:
                    print(
                        f"[PROGRESS] {agent} step {step}/{steps}, {time.perf_counter() - t_agent:.0f} s, "
                        f"episodes ended {len(episode_returns)}",
                        flush=True,
                    )
                if sim.is_rendering:
                    time.sleep(max(0.0, step_time - (time.perf_counter() - frame_start)))

            elapsed = time.perf_counter() - t_agent
            checks = [
                "finite",
                "shapes",
                "start pose after reset",
                "stem base in the spawn area",
                "target_offset <= 1",
                "applied_step <= 1 (clamp free)",
                "stem points within their arc length",
                "stem_base constant within an episode",
                "target_pos constant within an episode",
                "reward ranges",
                "reset only with a done flag",
            ]
            if agent == "zero":
                checks += ["zero agent: only the time out, at step 469", "zero agent: tool tip holds",
                           "zero agent: penalties 0"]  # fmt: skip
            if timed_out:
                checks.append("wall-clock limit")
            values = {
                "start pose after reset": f"{stats['start'] * 1e3:.2f} mm, {stats['angle']:.3f} deg",
                "target_offset <= 1": f"max {stats['offset']:.4f}",
                "applied_step <= 1 (clamp free)": f"max {stats['step_out']:.4f}",
                "stem points within their arc length": f"max excess {stats['arc'] * 1e3:.3f} mm",
                "zero agent: tool tip holds": f"max {stats['hold'] * 1e3:.3f} mm",
            }
            for check in checks:
                found = fails.get(check, [])
                info = values.get(check, "")
                if found:
                    info = f"{info} {len(found)} failing steps, first: {found[:3]}".strip()
                results[f"{agent}: {check}"] = (not found, info or "ok")
            results[f"{agent}: tool_tip_rot unit and orthogonal"] = (
                stats["unit"] <= UNIT_TOL,
                f"max deviation {stats['unit']:.1e}",
            )
            if timed_out:
                # a row that already failed stays FAIL; the others are incomplete
                results_not_run.update(k for k, (ok, _) in results.items() if k.startswith(f"{agent}: ") and ok)
            mean_return = sum(episode_returns) / len(episode_returns) if episode_returns else float("nan")
            print(
                f"[INFO] {agent} agent: {steps_run} of {steps} steps in {elapsed:.0f} s "
                f"({elapsed / max(steps_run, 1) * 1e3:.0f} ms per step), episodes ended {len(episode_returns)}, "
                f"mean return {mean_return:.2f}; terminations (env-steps): "
                + ", ".join(f"{k} {v}" for k, v in fired.items()),
                flush=True,
            )
            if timed_out:
                break

        print(f"\n=== check_push_agents ({n} envs, agents {' '.join(args_cli.agents)}) ===")
        for name, (ok, info) in results.items():
            status = "NOT RUN" if name in results_not_run else "PASS" if ok else "FAIL"
            print(f"[{status}] {name}: {info}")
        print(f"total {time.perf_counter() - t_start:.0f} s")
        passed = all(ok for ok, _ in results.values()) and not results_not_run
        print("=== all passed ===" if passed else "=== FAILED ===")
        env.close()


if __name__ == "__main__":
    main()
