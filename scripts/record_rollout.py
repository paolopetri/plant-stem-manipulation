"""Characterization check for refactors of simulator code: record a fixed-seed rollout, compare two recordings.

The `check_*.py` scripts test against bands, so a refactor could change behaviour inside them unnoticed. This
script records a short rollout of `StemManip-Push-Position-FR3-v0` before the refactor and compares it with one
recorded after it:
- `record <out>`: seeds the env (`--seed`, also the seed of every reset) and runs three rollouts: `--steps` policy
  steps of zero actions, `--steps` of uniform random actions in [-1, 1] (a torch generator seeded with `--seed`),
  and a scripted push (stem base fixed at `PUSH_STEM_BASE`, the slot base along `PUSH_PLAN`: twice behind the stem,
  down, then through the stem axis at full speed). The push makes the contact force, the contact-force and
  curvature penalties non-zero and `curvature_limit` and `contact_force_limit` fire; if one of these stays 0 / never
  fires, `record` fails (exit status 1) naming it and saves nothing. It saves every policy
  observation term (at the reset and after each step), every reward term (weighted, before x dt), every termination
  term, and the raw robot-stem contact force and the largest stem curvature (after the step) per step and env, with
  the settings and the commit (`torch.save`).
- `compare <a> <b>`: prints one row per term with its max absolute difference: PASS if within the tolerance, FAIL
  naming the term otherwise; a term missing in one recording, another shape, or other settings (task, envs, steps,
  seed, device, push rollout length) are FAIL. Exit status 0 if all passed, 1 otherwise. No simulator needed.
  `--rename old=new` (repeatable) compares a term renamed by the refactor: the baseline's `old` term is compared
  with the other recording's `new` term; how names match and what is an error (exit status 1):
  `rollout_compare.compare`.

Tolerance (user, 2026-10-14): bit-exact on the same device; GPU and CPU recordings differ. Without `--tol`, the
contact-sensor force terms (the raw robot-stem contact force and the contact-force reward, in every rollout) pass
within a float32 noise floor (user, 2026-10-15) and their rows say "float32 noise" when non-zero: the floor, its
reason and the terms are in `rollout_compare.CONTACT_FORCE_NOISE`. An explicit `--tol` applies to all terms, the
contact force included: a refactor that legitimately changes the floating-point order may pass `--tol` (e.g. 1e-6)
and states the reason in its PR.

Refactor workflow: recordings go under `logs/` (not committed); the baseline is recorded right before the refactor,
in the branch's worktree (check out the base commit there first, here `main`), not in the main checkout, so no
golden file goes stale.
    git switch --detach main
    uv run --extra isaacsim python scripts/record_rollout.py record logs/rollout/main.pt
    git switch <refactor branch>
    uv run --extra isaacsim python scripts/record_rollout.py record logs/rollout/refactor.pt
    uv run python scripts/record_rollout.py compare logs/rollout/main.pt logs/rollout/refactor.pt
    # if the refactor renamed a term, e.g. the observation slot_base_pos to slot_base_position:
    uv run python scripts/record_rollout.py compare logs/rollout/main.pt logs/rollout/refactor.pt \
        --rename slot_base_pos=slot_base_position

Recording takes about 1 min with the defaults (4 envs, 50 steps).
"""

import argparse
import sys
import time
from pathlib import Path

TASK = "StemManip-Push-Position-FR3-v0"
# push rollout: stem base (world x, y) [m], as in check_push_terms.py; slot-base waypoints relative to the stem base
# (robot base frame) [m] and the policy steps spent on each. Two pushes, each from behind the stem at the start
# height, down, then toward 0.15 m beyond the stem axis at full speed: at 0.15 m height the stem bends until
# `curvature_limit` fires and resets the envs; at 0.10 m (stiffer, the curvature within the curvature budget) the
# contact force passes its limit and `contact_force_limit` fires
PUSH_STEM_BASE = (0.65, 0.0)
PUSH_PLAN = (
    ((-0.20, 0.0, 0.50), 70),
    ((-0.20, 0.0, 0.15), 180),
    ((0.15, 0.0, 0.15), 110),
    ((-0.20, 0.0, 0.50), 70),
    ((-0.20, 0.0, 0.10), 200),
    ((0.15, 0.0, 0.10), 130),
)
PUSH_STEPS = sum(steps for _, steps in PUSH_PLAN)
# push terms that must be non-zero / fire at least once, or the push no longer exercises the damage limits
PUSH_REQUIRED = (
    "push/reward/contact_force",
    "push/reward/curvature",
    "push/termination/curvature_limit",
    "push/termination/contact_force_limit",
)


def rename(text: str) -> tuple[str, str]:
    """`old=new` -> `(old, new)`, for `--rename`."""
    old, sep, new = text.partition("=")
    if not (sep and old and new):
        raise argparse.ArgumentTypeError(f"expected old=new, got {text!r}")
    return old, new


parser = argparse.ArgumentParser(description="Record a fixed-seed rollout of the push task, or compare two.")
commands = parser.add_subparsers(dest="command", required=True)
record_parser = commands.add_parser("record", help="Record a rollout to a file.")
record_parser.add_argument("out", type=Path, help="Output file, e.g. logs/rollout/main.pt.")
record_parser.add_argument("--num_envs", type=int, default=4, help="Number of envs.")
record_parser.add_argument("--steps", type=int, default=50, help="Policy steps of the zero and random rollouts.")
record_parser.add_argument("--seed", type=int, default=42, help="Seed of the env and of the random actions.")
compare_parser = commands.add_parser("compare", help="Compare two recordings.")
compare_parser.add_argument("a", type=Path, help="Baseline recording.")
compare_parser.add_argument("b", type=Path, help="Recording to compare with the baseline.")
compare_parser.add_argument(
    "--tol",
    type=float,
    default=None,
    help="Largest absolute difference that passes, for every term (default: the rule in the docstring).",
)
compare_parser.add_argument(
    "--rename",
    type=rename,
    action="append",
    default=[],
    metavar="OLD=NEW",
    help="Term renamed between the recordings (repeatable), e.g. slot_base_pos=slot_base_position.",
)

if sys.argv[1:2] == ["record"]:
    # imported only to record: importing isaaclab.app starts Kit's bootstrap
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(record_parser)
args_cli = parser.parse_args()

import torch  # noqa: E402

from stem_manip.utils import check_report, rollout, rollout_compare  # noqa: E402


def record() -> None:
    """Run the zero, random and push rollouts and save the recording."""
    import gymnasium as gym

    import stem_manip.tasks  # noqa: F401  (registers the task)
    from stem_manip.tasks.push_position import mdp
    from stem_manip.tasks.push_position.env_cfg import StemPushPositionEnvCfg
    from stem_manip.tasks.push_position.mdp.stem_state import stem_state

    env_cfg = StemPushPositionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    t_start = time.perf_counter()
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        n, device = env.num_envs, env.device
        action_dim = env.action_manager.total_action_dim
        slot_base_cfg = env.action_manager.get_term("slot_base").cfg
        slices = rollout.obs_term_slices(
            env.observation_manager.active_terms["policy"], env.observation_manager.group_obs_term_dim["policy"]
        )
        reward_names = env.reward_manager.active_terms
        term_names = env.termination_manager.active_terms
        # the contact sensor of the damage terms, from their cfg
        sensor_cfg = env.reward_manager.get_term_cfg("contact_force").params["sensor_cfg"]
        terms: dict[str, torch.Tensor] = {}

        def run(agent: str, steps: int, policy) -> None:
            """Reset with the seed, run `steps` policy steps of `policy(obs) -> action`, add the terms."""
            obs = [env.reset(seed=args_cli.seed)[0]["policy"].clone()]
            rewards, terminations = [], {name: [] for name in term_names}
            state = {"robot_contact_force": [], "max_curvature": []}
            for _ in range(steps):
                obs.append(env.step(policy(obs[-1]))[0]["policy"].clone())
                # weighted reward per term (before x dt); the manager has no public accessor for it
                rewards.append(env.reward_manager._step_reward.clone())
                for name in term_names:
                    terminations[name].append(env.termination_manager.get_term(name).clone())
                # raw damage quantities after the step (after the reset in envs that were reset)
                state["robot_contact_force"].append(mdp.robot_contact_force(env, sensor_cfg))
                state["max_curvature"].append(stem_state(env).curvature().amax(dim=-1))
            obs_all, reward_all = torch.stack(obs).cpu(), torch.stack(rewards).cpu()
            for name, part in slices.items():
                terms[f"{agent}/obs/{name}"] = obs_all[..., part]
            for i, name in enumerate(reward_names):
                terms[f"{agent}/reward/{name}"] = reward_all[..., i]
            for name in term_names:
                terms[f"{agent}/termination/{name}"] = torch.stack(terminations[name]).cpu()
            for name, values in state.items():
                terms[f"{agent}/state/{name}"] = torch.stack(values).cpu()

        for agent in ("zero", "random"):
            generator = torch.Generator(device=device).manual_seed(args_cli.seed)
            run(agent, args_cli.steps, lambda _obs, a=agent, g=generator: rollout.agent_action(a, n, action_dim, g))

        # push: the stem base at PUSH_STEM_BASE, the slot base along PUSH_PLAN (relative to the stem base).
        # The `spawn_stem` event cfg params are updated in place; this works because the event manager reads the
        # params at call time (each reset), not when the env is built.
        params = env.event_manager.get_term_cfg("spawn_stem").params
        params.update(x_range=(PUSH_STEM_BASE[0],) * 2, y_range=(PUSH_STEM_BASE[1],) * 2)
        plan = [(torch.tensor(offset, device=device), steps) for offset, steps in PUSH_PLAN]
        clock = {"step": 0}

        def push(obs: torch.Tensor) -> torch.Tensor:
            k, leg = clock["step"], 0
            while k >= plan[leg][1]:
                k, leg = k - plan[leg][1], leg + 1
            clock["step"] += 1
            # steering from the policy observations: valid because observation corruption is off
            # (`enable_corruption = False` in push_position env_cfg) and both terms are in the robot base frame
            goal = obs[:, slices["stem_base"]] + plan[leg][0]
            return rollout.toward(
                obs[:, slices["slot_base_pos"]], goal, slot_base_cfg.max_step, slot_base_cfg.max_step_change
            )

        run("push", PUSH_STEPS, push)
        missing = rollout_compare.never_nonzero(terms, PUSH_REQUIRED)
        if missing:
            print(f"[ERROR] the push rollout no longer exercises the damage limits, 0 / never fired: {missing}")
            env.close()
            sys.exit(1)

        settings = rollout_compare.Settings(TASK, n, args_cli.steps, args_cli.seed, str(device), PUSH_STEPS)
        meta = rollout_compare.meta(settings, env_cfg.git_commit)
        args_cli.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": meta, "terms": terms}, args_cli.out)
        print(f"[INFO] recorded {len(terms)} terms ({meta}) to {args_cli.out} in {time.perf_counter() - t_start:.0f} s")
        env.close()


def compare() -> None:
    """Print the rows of the two recordings and exit with the verdict."""
    a, b = (torch.load(path) for path in (args_cli.a, args_cli.b))
    tol = f"tol {args_cli.tol:g}" if args_cli.tol is not None else "default tol"
    print(f"=== record_rollout compare {args_cli.a} {args_cli.b} ({tol}) ===")
    try:
        rows = rollout_compare.compare(a, b, args_cli.tol, args_cli.rename)
    except ValueError as error:
        print(f"[ERROR] --rename: {error}")
        sys.exit(1)
    check_report.finish(rows)


if __name__ == "__main__":
    if args_cli.command == "record":
        record()
    else:
        compare()
