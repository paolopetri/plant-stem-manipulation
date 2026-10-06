# CLAUDE.md

Project instructions for the plant-stem manipulation project. General coding guidelines are in the user-level `~/.claude/CLAUDE.md`.

## Project

Semester project (ETH Zurich MSc RSC, hosted at AUTOLAB, UNIMORE Modena):
**Non-prehensile manipulation of flexible plant stems with reinforcement learning.**

A robot pushes (does not grasp) a flexible stem with a 3D-printed tool so that a selected point on the stem, e.g. where a leaf is attached, reaches a target pose. The stem must not be damaged. The stem's mechanical properties (stiffness, length, diameter, damping) are unknown and vary per plant, so they are domain-randomized during training.

The learned policy is a low-level skill. A future perception module will choose the target; that module is out of scope (unlike the stem-state perception below, which is in scope).

For the full motivation, research question, and scope, read `docs/project-proposal.md` before making modelling, reward, or task-design decisions. For routine coding tasks it is not needed.

## Stack

- Simulation: NVIDIA Isaac Sim
- RL framework: Isaac Lab (PPO)
- Reference robot: Franka Research 3. Actions are defined in end-effector space so the skill is not tied to one arm.
- End-effector: `fork_v2` (`assets/fr3/end_effectors/fork_v2/`). `fork` is the first design, kept but not used.
- Stem model: deformable linear object. Implemented: Newton cable (`cable`, M1) and a chain of rigid segments with bending/twist spring joints as a PhysX articulation (`chain`, M1b), which shares one solver with the robot. The supervisors chose `chain` for the training (2026-10-06); `cable` is kept for comparison. Stem models are kept modular (selectable by name, like the end-effectors). Background: `docs/TODO.md`, open questions (stem model, contact handling).
- Assets are managed as USD files.

## Task stages (in order)

1. Position control of the selected stem point
2. Full pose control (position + orientation) of the attached frame
3. Real-robot validation on an artificial plant, with the stem state estimated from cameras

Perception (camera-based stem-state estimate, with Alessio Caporali) runs in parallel with the first training runs. It was not in the original proposal, but the real-robot transfer is likely needed to answer the research question and gives the more interesting result (decision 2026-10-06). Stretch goal: distilling the policy into an end-to-end vision-based one.

## Key design constraints

- Contact is **non-prehensile**: intermittent, unilateral, with friction and sliding. Never assume the tool is attached to the stem.
- **No-damage constraint**: the working definition is a maximum curvature (minimum bending radius) on the stem. Treat it as a hard limit, enforced through termination and/or penalty.
- In simulation the policy observes the stem state directly; on the real robot that state comes from cameras, so the observations must stay within what the perception can deliver. Contact forces are not observed; they may be used in rewards and terminations (decision 2026-10-06).
- A model-based baseline (nominal parameters vs. oracle parameters) is planned where feasible.
- Many modelling choices are still being decided with the supervisors. Ask before committing to one.

## Verification in this project

Much of this code is simulation and RL, where unit tests don't cover everything. Use checks like these as success criteria:
- The environment launches and steps without errors (short headless run with few envs).
- Observation/action shapes and value ranges match expectations.
- Physics sanity: the stem sags under gravity, springs back after release, and deflects when pushed.
- Reward terms and terminations fire in the cases they should (e.g. curvature limit exceeded).

## Repo layout

Installable uv package `stem_manip` (Isaac Lab 3 external-project layout). Details: `docs/architecture.md`.
- `assets/`: source data and build pipelines (`fr3/` URDF -> USD, `stem/stem.yaml` = plant + `stem/<model>/<model>.yaml`). No Isaac Lab cfgs here.
- `src/stem_manip/assets/`: Isaac Lab configs pointing to `assets/` (`fr3.py`, `stem/` with one module per stem model, selected by name via `stem_model(name)`).
- `src/stem_manip/tasks/<task>/`: one package per task (env cfg, `mdp/`, `agents/`), registered via the `isaaclab.tasks` entry point.
- `src/stem_manip/utils/`: simulator-free helpers (e.g. stem curvature).
- `scripts/`: sanity checks (`check_*.py`). `tests/`: unit tests without the simulator.
- `docs/`: `architecture.md`, `TODO.md`, `notes/`.

Skeleton modules contain only a spec docstring (requirements, how to verify, TODO milestone) until implemented.

## Commands

Isaac Lab 3 source checkout expected at `../IsaacLab`.
- Setup: `uv sync --extra isaacsim --extra rsl-rl`
- List tasks: `uv run isaaclab list_envs`
- Unit tests: `uv run pytest`
- Simulation checks (headless by default, no `--headless` flag in Isaac Lab 3): `uv run --extra isaacsim python scripts/check_<name>.py`, e.g. `check_fr3.py`
- Train / play: added once the first task is registered (docs/TODO.md, M5).

## Git workflow

- `main` must always run. Never commit directly to `main`.
- One short-lived branch per unit of work: `feat/`, `fix/`, `exp/`, `refactor/`, `docs/` (e.g. `feat/asset-USD-management`).
- Make small, focused commits with messages in the imperative ("Add curvature termination").
- Merge via a pull request (squash-merge), then delete the branch.
- `exp/` branches may stay unmerged.
- Tag milestones, e.g. `v0.1-position-control`.
- No `Co-Authored-By` trailer or other AI attribution in commit messages or PR descriptions.

## Conventions

- Keep all parameters in versioned config files: randomization ranges, reward weights, stem properties, curvature limit. Do not hardcode them in scripts.
- Log the git commit hash with every training run.
- Never commit logs, checkpoints, videos, or `wandb/`. They are in `.gitignore`.
- Python code: type hints and docstrings on public functions; follow Isaac Lab's config-class style.

## Notes

- `docs/notes/YYYY-MM-DD.md`: one short, append-only log per work day, with sections Done / Decisions / Open questions (skip empty ones).
- `docs/TODO.md`: the only place for open items. When an item is done, delete it there and log it in today's note.
- After finishing a unit of work, add an entry to today's note and update `docs/TODO.md`.