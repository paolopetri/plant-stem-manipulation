"""Characterization check for refactors of simulator code: record a fixed-seed rollout, compare two recordings.

The `check_*.py` scripts test against bands, so a refactor could change behaviour inside them unnoticed. This
script records a short rollout of `StemManip-Push-Position-FR3-v0` before the refactor and compares it with one
recorded after it:
- `record <out>`: seeds the env (`--seed`, also the seed of every reset) and runs two rollouts of `--steps` policy
  steps: zero actions, then uniform random actions in [-1, 1] (a torch generator seeded with `--seed`). It saves
  every policy observation term (at the reset and after each step), every reward term (weighted, before x dt) and
  every termination term per step and env, together with the settings and the commit, with `torch.save`.
- `compare <a> <b>`: prints one row per term with its max absolute difference: PASS if within `--tol`, FAIL
  naming the term otherwise; a term missing in one recording, another shape, or other settings (task, envs, steps,
  seed, device) are FAIL. Exit status 0 if all passed, 1 otherwise. No simulator needed.

Default tolerance 0 (bit-exact): two recordings with the same settings on the same machine were bit-identical on
the GPU and on the CPU, also over 500 steps with resets (issue #52). Compare recordings of one device: GPU and CPU
recordings differ. Neither rollout pushes the stem (contact and curvature terms stay 0), so a change that only
acts in contact is not seen.

Refactor workflow: recordings go under `logs/` (not committed); the baseline is recorded on `main` right before the
refactor, so no golden file goes stale.
    git switch main
    uv run --extra isaacsim python scripts/record_rollout.py record logs/rollout/main.pt
    git switch <refactor branch>
    uv run --extra isaacsim python scripts/record_rollout.py record logs/rollout/refactor.pt
    uv run python scripts/record_rollout.py compare logs/rollout/main.pt logs/rollout/refactor.pt

Recording takes about 15 s with the defaults (4 envs, 50 steps), 100 s with `--steps 500`.
"""

import argparse
import sys
import time
from pathlib import Path

TASK = "StemManip-Push-Position-FR3-v0"
AGENTS = ("zero", "random")
ACTION_DIM = 6

parser = argparse.ArgumentParser(description="Record a fixed-seed rollout of the push task, or compare two.")
commands = parser.add_subparsers(dest="command", required=True)
record_parser = commands.add_parser("record", help="Record a rollout to a file.")
record_parser.add_argument("out", type=Path, help="Output file, e.g. logs/rollout/main.pt.")
record_parser.add_argument("--num_envs", type=int, default=4, help="Number of envs.")
record_parser.add_argument("--steps", type=int, default=50, help="Policy steps per rollout.")
record_parser.add_argument("--seed", type=int, default=42, help="Seed of the env and of the random actions.")
compare_parser = commands.add_parser("compare", help="Compare two recordings.")
compare_parser.add_argument("a", type=Path, help="Baseline recording.")
compare_parser.add_argument("b", type=Path, help="Recording to compare with the baseline.")
compare_parser.add_argument("--tol", type=float, default=0.0, help="Largest absolute difference that passes.")
if sys.argv[1:2] == ["record"]:
    # imported only to record: importing isaaclab.app starts Kit's bootstrap
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(record_parser)
args_cli = parser.parse_args()

import torch  # noqa: E402

from stem_manip.utils import check_report, rollout_compare  # noqa: E402


def record() -> None:
    """Run the zero and the random rollout and save the recording."""
    import math

    import gymnasium as gym

    import stem_manip.tasks  # noqa: F401  (registers the task)
    from stem_manip.tasks.push_position.env_cfg import StemPushPositionEnvCfg

    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    t_start = time.perf_counter()
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        n, device = env.num_envs, env.device
        obs_names = env.observation_manager.active_terms["policy"]
        obs_dims = env.observation_manager.group_obs_term_dim["policy"]
        slices, start = {}, 0
        for name, dims in zip(obs_names, obs_dims, strict=True):
            slices[name] = slice(start, start + math.prod(dims))
            start += math.prod(dims)
        reward_names = env.reward_manager.active_terms
        term_names = env.termination_manager.active_terms
        terms: dict[str, torch.Tensor] = {}
        for agent in AGENTS:
            obs = [env.reset(seed=args_cli.seed)[0]["policy"].clone()]
            rewards, terminations = [], {name: [] for name in term_names}
            generator = torch.Generator(device=device).manual_seed(args_cli.seed)
            for _ in range(args_cli.steps):
                if agent == "zero":
                    action = torch.zeros(n, ACTION_DIM, device=device)
                else:
                    action = 2 * torch.rand(n, ACTION_DIM, device=device, generator=generator) - 1
                obs_dict = env.step(action)[0]
                obs.append(obs_dict["policy"].clone())
                rewards.append(env.reward_manager._step_reward.clone())  # weighted, per term (before x dt)
                for name in term_names:
                    terminations[name].append(env.termination_manager.get_term(name).clone())
            obs_all, reward_all = torch.stack(obs).cpu(), torch.stack(rewards).cpu()
            for name in obs_names:
                terms[f"{agent}/obs/{name}"] = obs_all[..., slices[name]]
            for i, name in enumerate(reward_names):
                terms[f"{agent}/reward/{name}"] = reward_all[..., i]
            for name in term_names:
                terms[f"{agent}/termination/{name}"] = torch.stack(terminations[name]).cpu()
        meta = {
            "task": TASK,
            "num_envs": n,
            "steps": args_cli.steps,
            "seed": args_cli.seed,
            "device": str(device),
            "commit": env_cfg.git_commit,
        }
        args_cli.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": meta, "terms": terms}, args_cli.out)
        print(f"[INFO] recorded {len(terms)} terms ({meta}) to {args_cli.out} in {time.perf_counter() - t_start:.0f} s")
        env.close()


def compare() -> None:
    """Print the rows of the two recordings and exit with the verdict."""
    a, b = (torch.load(path) for path in (args_cli.a, args_cli.b))
    print(f"=== record_rollout compare {args_cli.a} {args_cli.b} (tol {args_cli.tol:g}) ===")
    check_report.finish(rollout_compare.compare(a, b, args_cli.tol))


if __name__ == "__main__":
    if args_cli.command == "record":
        record()
    else:
        compare()
