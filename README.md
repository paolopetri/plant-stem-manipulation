# Plant stem manipulation

Non-prehensile manipulation of flexible plant stems with reinforcement learning (Isaac Lab 3, FR3 + fork).
Semester project, ETH Zurich MSc RSC, hosted at AUTOLAB (UNIMORE, Modena).

- Architecture and where things go: [docs/architecture.md](docs/architecture.md)
- Roadmap and open items: [docs/TODO.md](docs/TODO.md)
- FR3 + end-effector asset pipeline: [assets/fr3/README.md](assets/fr3/README.md)

## Setup

Requires an Isaac Lab 3 source checkout next to this repo (`../IsaacLab`, see `[tool.uv.sources]` in
`pyproject.toml`) and [uv](https://docs.astral.sh/uv/). Check out `isaac-sim/IsaacLab` at the commit recorded in
`ISAACLAB_COMMIT` (CI uses the same one).

```bash
uv sync --extra isaacsim --extra rsl-rl --extra wandb --extra video
```

This creates `.venv/` with Isaac Lab (editable, from `../IsaacLab`) and this package (`stem_manip`, editable).

Install the pre-commit hook (ruff check and ruff format on each `git commit`):

```bash
uv run pre-commit install
```

## Commands

All commands are run from the repo root. Each script's module docstring is its manual (what it does, its checks
and their pass criteria); `--help` lists its options.

```bash
uv run isaaclab list_envs                               # lists registered tasks, including this project's
uv run pytest                                           # unit tests (no simulator)
uv run --extra isaacsim python scripts/check_fr3.py     # FR3 + end-effector asset: pass/fail
uv run --extra isaacsim python scripts/check_stem.py     # chain stem model alone: pass/fail
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick   # cable stem: accuracy of one solver setting
uv run --extra isaacsim python scripts/check_contact.py --test slot       # FR3 + fork push the chain stem: pass/fail
uv run --extra isaacsim python scripts/sweep_impedance.py                 # impedance controller study (fork + stem)
uv run --extra isaacsim python scripts/check_push_env.py      # push task: controller and action limits, pass/fail
uv run --extra isaacsim python scripts/check_push_stem.py     # push task: the action pushes the stem, contact force
uv run --extra isaacsim python scripts/sweep_action_poses.py  # push task: following over the workspace
uv run --extra isaacsim python scripts/check_push_obs.py      # push task: target command and stem observations
uv run --extra isaacsim python scripts/check_push_terms.py    # push task: rewards and terminations
uv run --extra isaacsim python scripts/check_push_agents.py   # push task: zero / random agent, shapes and ranges
uv run --extra isaacsim python scripts/record_rollout.py record logs/rollout/<name>.pt   # fixed-seed rollout
uv run python scripts/record_rollout.py compare logs/rollout/main.pt logs/rollout/<name>.pt   # pass/fail
uv run --extra isaacsim isaaclab zero_agent --task StemManip-Push-Position-FR3-v0 --num_envs 4 --max_steps 700
uv run --extra isaacsim isaaclab random_agent --task StemManip-Push-Position-FR3-v0 --num_envs 4 --max_steps 700
```

The zero / random agents must be started through the `isaaclab` CLI as above (Isaac Lab's
`scripts/environments/zero_agent.py` run directly does not register this project's tasks), and they need
`--max_steps`: without it they run until the process is killed.
Training: see "Push task: training" below.

Before a refactor of simulator code, record a rollout on `main` and compare it with one recorded after the
refactor: workflow in the docstring of `scripts/record_rollout.py`.

Every `scripts/check_*.py` ends with `[PASS]`, `[FAIL]` or `[NOT RUN]` per row and a verdict line
(`=== all passed ===` or `=== FAILED ===`). The exit status carries the verdict: 0 if all passed, 1 on any FAIL or
NOT RUN (a crash is non-zero too).

## Watching the checks

Simulation scripts run without a window unless a viewer is requested: `--viz kit` (Isaac Sim's window, PhysX
scenes: the `chain` stem and the push task) or `--viz newton_gl` (Newton's viewer, both stem models).
`--slow_motion S` plays S times slower than real time. Kit's very first start takes several minutes without output
(window "not responding"); later starts take seconds.

```bash
uv run --extra isaacsim python scripts/check_stem.py --viz kit --slow_motion 5
uv run --extra isaacsim python scripts/sweep_stem_solver.py --test kick --viz newton_gl --slow_motion 5 --steps 200
uv run --extra isaacsim python scripts/check_contact.py --test slot_far --viz kit --slow_motion 3
uv run --extra isaacsim python scripts/check_push_env.py --num_envs 1 --viz kit --slow_motion 3
uv run --extra isaacsim python scripts/check_push_stem.py --num_envs 1 --viz kit --slow_motion 3
uv run --extra isaacsim python scripts/check_push_obs.py --num_envs 4 --viz kit --slow_motion 3
uv run --extra isaacsim python scripts/check_push_terms.py --phases push --viz kit --slow_motion 1
uv run --extra isaacsim python scripts/check_push_agents.py --agents random --num_envs 4 --viz kit --slow_motion 2
```

For reproducible clips (presentations), `check_contact.py --video` renders offscreen from a fixed camera and writes
`videos/check_contact_<test>_<NNNN>.mp4` (Isaac Lab's `VideoRecorder`), `--slow_motion` times slower than real
time. Rendering makes the run slower (several minutes):

```bash
uv run --extra isaacsim --extra video python scripts/check_contact.py --test slot_far --video --slow_motion 3
```

`sweep_stem_solver.py` replays its test until the window is closed; the viewer's left panel pauses and resumes it.
What each scene shows (markers, phases): the script's docstring.

## Push task: training

RSL-RL PPO (`agents/rsl_rl_ppo_cfg.py`), logged to wandb (project `stem-manip`; once: `uv run wandb login` in a
terminal) and to local tensorboard files under `logs/rsl_rl/stem_push_position/<date>_<run name>/`. Every run logs
this repo's commit (`git_commit` in `params/env.yaml` and the wandb config; `-dirty` = uncommitted changes).
Iteration count and PPO settings: `agents/rsl_rl_ppo_cfg.py`. Speed (2026-10-10 benchmark): about 5.3 s per iteration
with 4096 envs (6 GB GPU memory), 2.8 / 2.9 / 3.4 s with 512 / 1024 / 2048 envs. Recording videos during training (`--video`) doubles the iteration time, so videos are
recorded afterwards with `play`.

```bash
uv run --extra isaacsim isaaclab train --task StemManip-Push-Position-FR3-v0 --num_envs 4096 --run_name <name>
uv run python scripts/summarize_run.py logs/rsl_rl/stem_push_position/<date>_<name>     # compact table of the run
# evaluate a checkpoint with the mean action, as play runs it: how the episodes end, final errors, fork rotation
# (one episode per env, about 4 min with the default envs; per-episode values in <run dir>/eval/)
uv run --extra isaacsim python scripts/evaluate_policy.py logs/rsl_rl/stem_push_position/<date>_<name>/model_<iteration>.pt
# play the last checkpoint and record 2 episodes of env 0 (writes <run dir>/videos/play/), then put it on wandb
uv run --extra isaacsim --extra video isaaclab play --task StemManip-Push-Position-FR3-v0 --num_envs 4 \
    --checkpoint logs/rsl_rl/stem_push_position/<date>_<name>/model_<last iteration>.pt --video --video_length 938
uv run python tools/autotrain/upload_video.py logs/autotrain/<name>.out logs/rsl_rl/stem_push_position/<date>_<name>
# watch a policy in the viewer
uv run --extra isaacsim isaaclab play --task StemManip-Push-Position-FR3-v0 --num_envs 4 --checkpoint <model.pt> --viz kit
```

Play from a chosen start pose (evaluation outside the training's start-pose table): `isaaclab play` passes Hydra
overrides to the env cfg, so the reset event's `poses` (one or more arm joint vectors, rad) and `joint_noise` can be
set on the command line. `scripts/start_poses.py pose` prints the `poses` line: a held-out pose (`a_above_tip`,
`b_side_90`, `c_low`) as committed, or a new pose beside the stem solved with IK (slot-base height in m, azimuth
about the stem axis in deg; it also prints the joint margin and arm clearance, which the table requires to be at
least 0.5 rad and 5 cm). `joint_noise=0.0` starts exactly at the pose.

```bash
uv run --extra isaacsim python scripts/start_poses.py pose --held_out b_side_90
uv run --extra isaacsim python scripts/start_poses.py pose --height 0.25 --azimuth 30
uv run --extra isaacsim --extra video isaaclab play --task StemManip-Push-Position-FR3-v0 --num_envs 1 \
    --checkpoint <model.pt> --video --video_length 938 \
    'env.events.start_pose.params.poses=[[-0.515,0.378,0.096,-1.873,-0.046,2.249,-1.183]]' \
    env.events.start_pose.params.joint_noise=0.0
```

`upload_video.py` reads the wandb run id from the train output, so it needs the run's output saved to a file
(`... > logs/autotrain/<name>.out 2>&1`). The output is large (about 120 harmless PhysX warnings per env at
startup, "Failed to find rigid body": Isaac Lab's PhysX manager creates a rigid-body view over every prim, `/**`).

### Autotrain loop (weekend runs)

`tools/autotrain/`: a bash driver waits on the runs and calls a fresh Claude agent (`claude -p`) for each decision:
analyse the last run, change one thing (tuning or a proven bug fix, rules in `tools/autotrain/prompt.md`), commit,
launch the next run (train, then a play video uploaded to wandb). Journal: `docs/experiments/`.

```bash
# deadline, branch (must be checked out) and wandb group are required; without them the driver prints a usage message
AUTOTRAIN_DEADLINE="YYYY-MM-DD HH:MM" AUTOTRAIN_BRANCH=exp/<name> AUTOTRAIN_GROUP=<wandb group> \
    screen -S autotrain tools/autotrain/driver.sh   # detach with Ctrl-a d, back: screen -r autotrain
touch logs/autotrain/STOP                       # stop: the alive run finishes, the agent writes the summary
tail -f logs/autotrain/driver.log               # what the driver does
```
