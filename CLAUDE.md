# CLAUDE.md

Project instructions for the plant-stem manipulation project. General coding guidelines are in the user-level `~/.claude/CLAUDE.md`.

## Project

Semester project (ETH Zurich MSc RSC, hosted at AUTOLAB, UNIMORE Modena): **non-prehensile manipulation of flexible plant stems with reinforcement learning.** A robot pushes a flexible stem with a 3D-printed tool so that a selected point on the stem reaches a target pose, without damaging the stem; the stem's properties are domain-randomized.

`docs/project.md` is the project description: motivation, research question, stages, scope, design constraints, people. Read it before modelling, reward or task-design decisions. The original proposal (`docs/proposal-original.md`) is frozen; where it differs, `docs/project.md` wins.

## Stack

- Simulation: NVIDIA Isaac Sim
- RL framework: Isaac Lab (PPO)
- Reference robot: Franka Research 3. Actions are defined in end-effector space so the skill is not tied to one arm.
- End-effector: `fork_v2` (`assets/fr3/end_effectors/fork_v2/`). `fork` is the first design, kept but not used.
- Stem model: deformable linear object. Implemented: Newton cable (`cable`, M1) and a chain of rigid segments with bending/twist spring joints as a PhysX articulation (`chain`, M1b), which shares one solver with the robot. The supervisors chose `chain` for the training (2026-10-06); `cable` is kept for comparison. Stem models are kept modular (selectable by name, like the end-effectors). Background on the parked `cable` path (stem model, contact handling): parked issues #96–#99.
- Assets are managed as USD files.

## Key design constraints

Full list in `docs/project.md`. The ones that most often matter in code:
- Contact is **non-prehensile**. Never assume the tool is attached to the stem.
- **No damage** is a hard limit, enforced through termination and/or penalty.
- Observations stay within what the camera-based perception can deliver. Contact forces are never observed; they may be used in rewards and terminations.
- Many modelling choices are still being decided with the supervisors. Ask before committing to one.

## Repo layout

Installable uv package `stem_manip` (Isaac Lab 3 external-project layout). Layers, stem models and where new code goes: `docs/architecture.md`.

## Commands

Isaac Lab 3 source checkout expected at `../IsaacLab`.
- Setup: `uv sync --extra isaacsim --extra rsl-rl --extra wandb --extra video`
- List tasks: `uv run isaaclab list_envs`
- Unit tests: `uv run pytest`
- Simulation checks (headless by default, no `--headless` flag in Isaac Lab 3): `uv run --extra isaacsim python scripts/check_<name>.py`, e.g. `check_fr3.py`
- Train / play / run summary / autotrain loop: README, section "Push task: training".

## Git workflow

- `main` must always run. Never commit directly to `main`.
- One short-lived branch per unit of work: `feat/`, `fix/`, `exp/`, `refactor/`, `docs/` (e.g. `feat/asset-USD-management`).
- Make small, focused commits with messages in the imperative ("Add curvature termination").
- Merge via a pull request, then delete the branch. Squash-merge by default; use a merge commit when individual commits must survive on main: an /implement-spec integration branch, or a commit named in .git-blame-ignore-revs.
- `exp/` branches may stay unmerged.
- Tag milestones, e.g. `v0.1-position-control`.
- Never commit logs, checkpoints, videos, or `wandb/`. They are in `.gitignore`.

## Coding standards

Before writing or reviewing code, read `CODING_STANDARDS.md`: decision tags, single source of truth, Python style, verification criteria.

## Notes

- `docs/notes/YYYY-MM-DD.md`: one short, append-only log per work day, with sections Done / Decisions / Open questions (skip empty ones). A decision that gets an ADR links it (`docs/adr/NNNN-...`) in the day's Decisions section.
- `docs/TODO.md`: high-level open items and discussion (direction, open questions, parked topics). Concrete, actionable work items are GitHub issues. When a TODO item is done or turned into issues, delete it there and log it in today's note.
- After finishing a unit of work, add an entry to today's note and update `docs/TODO.md`. Both go into the unit's own PR, not a separate notes PR (user, 2026-10-15). In orchestrated or parallel work the orchestrator writes the day's note, not each agent.

## Agent skills

- `docs/agents/workflow.md`: how new work goes from decision to PR, which skills the user starts, context and model choice.
- `docs/agents/issue-tracker.md`: GitHub issues via `gh`; every branch has an issue.
- `docs/agents/triage-labels.md`: label vocabulary.
- `docs/agents/domain.md`: domain docs, single-context: `GLOSSARY.md` and `docs/adr/` at the repo root.
