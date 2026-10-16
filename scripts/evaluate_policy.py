"""Evaluate a checkpoint of `StemManip-Push-Position-FR3-v0` with the mean action, as `isaaclab play` runs it.

Loads the policy the way Isaac Lab's `play` does (RSL-RL runner, `get_inference_policy`: the mean action, no
sampling; env seed = the agent cfg's seed), runs `--num_envs` envs and counts the first episode of every env (all envs
start together, so short episodes are not over-represented). Prints how the episodes end (time-out vs. each early
termination term), the final `position_error` / `height_error` (all episodes and time-out episodes only) and the
fork rotation (slot-base orientation vs. its orientation at the reset, largest and final), the contact point (over the
episodes with fork contact, from the stem metrics) and the share ended by arm contact, and saves the per-episode values
and the summary as JSON to `<run dir>/eval/<checkpoint>.json`.

`--start_pose NAME` starts every episode exactly at the held-out start pose `NAME` (`env_cfg.EVAL_START_POSES`, no
joint noise) instead of the training's start-pose table, and saves to `<run dir>/eval/<checkpoint>_<NAME>.json`.

Final values are those of the last step before the episode ends (user, 2026-10-15): the env resets inside the step
and recomputes the command metrics afterwards, so they are read before each step, the same values training logs as
`Metrics/...`. The contact point and fork contact are thus those of the episode without its last policy step, as
training logs them (`mdp/metrics.py`).

Usage:
    uv run --extra isaacsim python scripts/evaluate_policy.py <run dir>/model_<iteration>.pt [--num_envs 1024] \
        [--start_pose a_above_tip|b_side_90|c_low]
Summary: `stem_manip.utils.policy_eval` (tests: `tests/test_policy_eval.py`).
"""

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

TASK = "StemManip-Push-Position-FR3-v0"
COMMAND = "stem_target"
METRICS = "stem_metrics"

parser = argparse.ArgumentParser(description="Evaluate a checkpoint of the push task with the mean action.")
parser.add_argument("checkpoint", type=Path, help="Checkpoint, e.g. logs/rsl_rl/stem_push_position/<run>/model_749.pt.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of envs = number of evaluated episodes.")
parser.add_argument(
    "--start_pose", default=None, help="Held-out start pose (env_cfg.EVAL_START_POSES) instead of the table."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import torch  # noqa: E402

from stem_manip.utils import policy_eval  # noqa: E402
from stem_manip.utils.quat import rotation_angle  # noqa: E402


def evaluate() -> None:
    """Run the first episode of every env with the mean action, print and save the summary."""
    import gymnasium as gym

    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )

    import stem_manip.tasks  # noqa: F401  (registers the task)
    from stem_manip.tasks.push_position.agents.rsl_rl_ppo_cfg import StemPushPositionPPORunnerCfg
    from stem_manip.tasks.push_position.env_cfg import StemPushPositionEnvCfg, use_held_out_start_pose
    from stem_manip.tasks.push_position.mdp.slot_base import slot_base

    env_cfg = StemPushPositionEnvCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(StemPushPositionPPORunnerCfg(), check_rsl_rl_version())
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    env_cfg.seed = agent_cfg.seed  # as play
    if args_cli.start_pose is not None:
        use_held_out_start_pose(env_cfg, args_cli.start_pose)
    t_start = time.perf_counter()
    with launch_simulation(env_cfg, args_cli):
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(str(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        n, device = base.num_envs, base.device
        command = base.command_manager.get_term(COMMAND)
        stem_metrics = base.command_manager.get_term(METRICS)
        terms = base.termination_manager.active_terms
        (time_out,) = [name for name in terms if base.termination_manager.get_term_cfg(name).time_out]

        obs = env.get_observations()
        start_quat = slot_base(base).pose()[1].clone()  # every env is at its first reset
        max_rotation = torch.zeros(n, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        record = {
            "ended_by": {name: torch.zeros(n, dtype=torch.bool, device=device) for name in terms},
            "length_s": torch.zeros(n, device=device),
            "position_error": torch.zeros(n, device=device),
            "height_error": torch.zeros(n, device=device),
            "max_rotation": torch.zeros(n, device=device),
            "final_rotation": torch.zeros(n, device=device),
            "fork_contact": torch.zeros(n, dtype=torch.bool, device=device),
            "contact_point": torch.zeros(n, device=device),
        }
        with torch.inference_mode():
            while not done.all():
                # the state of the last step: the step below resets the envs whose episode ends
                rotation = rotation_angle(start_quat, slot_base(base).pose()[1])
                max_rotation = torch.maximum(max_rotation, rotation)
                before = {
                    "length_s": (base.episode_length_buf + 1) * base.step_dt,
                    "position_error": command.metrics["position_error"].clone(),
                    "height_error": command.metrics["height_error"].clone(),
                    "max_rotation": max_rotation.clone(),
                    "final_rotation": rotation,
                    "fork_contact": stem_metrics.metrics["fork_contact"] > 0,
                    "contact_point": stem_metrics.metrics["contact_point"].clone(),
                }
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                ended = dones.bool() & ~done
                for name in terms:
                    record["ended_by"][name][ended] = base.termination_manager.get_term(name)[ended]
                for key, value in before.items():
                    record[key][ended] = value[ended]
                done |= ended

        episodes = policy_eval.Episodes(
            ended_by={name: fired.cpu() for name, fired in record["ended_by"].items()},
            **{key: value.cpu() for key, value in record.items() if key != "ended_by"},
        )
        summary = policy_eval.summarize(episodes, time_out, command.cfg.success_distance)
        meta = {
            "checkpoint": str(args_cli.checkpoint.resolve()),
            "git_commit": env_cfg.git_commit,
            "num_envs": n,
            "seed": env_cfg.seed,
            "start_pose": args_cli.start_pose or "table",
            "success_distance": command.cfg.success_distance,
        }
        print(f"[INFO] {TASK}, {meta['checkpoint']}, commit {meta['git_commit']}, seed {meta['seed']}")
        print(f"[INFO] start pose: {meta['start_pose']}")
        print(format_summary(summary))
        suffix = f"_{args_cli.start_pose}" if args_cli.start_pose else ""
        out = args_cli.checkpoint.resolve().parent / "eval" / f"{args_cli.checkpoint.stem}{suffix}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = {"meta": meta, "summary": summary, "episodes": dataclasses.asdict(episodes)}
        out.write_text(json.dumps(result, indent=1, default=torch.Tensor.tolist))
        print(f"[INFO] saved to {out} ({time.perf_counter() - t_start:.0f} s)")
        env.close()


def format_summary(summary: dict[str, float | int]) -> str:
    """One row per value: shares in %, errors in mm, angles in degrees, lengths in s."""
    rows = []
    for key, value in summary.items():
        if key.startswith("ended_by/") or key.endswith(("/reached", "/fork_contact")):
            text = f"{100 * value:6.1f} %"
        elif "_error/" in key or "contact_point/" in key:
            text = f"{1000 * value:6.1f} mm"
        elif "_rotation/" in key:
            text = f"{math.degrees(value):6.1f} deg"
        elif key.startswith("length_s"):
            text = f"{value:6.2f} s"
        else:
            text = f"{value:6d}"
        rows.append(f"  {key:<34} {text}")
    return "\n".join(rows)


if __name__ == "__main__":
    evaluate()
