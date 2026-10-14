# Coding standards

The rules for writing and reviewing code in this repo; `/code-review` checks a diff against them. General guidelines are in the user-level `~/.claude/CLAUDE.md`.

## Decisions and history

- A decided value carries a short tag `(user, YYYY-MM-DD)` and nothing about how it got there. History lives in git, `docs/notes/` and ADRs (`docs/adr/`).
  - `# Start pose (user, 2026-10-09): ...`, not `# Start pose (user, 2026-10-09; was (0.30, 0, 0.55) m, 2026-10-06): ...`
  - Also history: "Step 2", "Implemented (M4, part 1)".
- Docstrings and comments say what the code is and why. Where the reason is long, link it (daily note, write-up, journal, ADR).
- Code points at open work by issue number (`#42`), never into `docs/TODO.md`.

## Single source of truth

- Every parameter lives in a versioned config file: randomization ranges, reward weights, stem properties, curvature limit. Scripts read it from there.
- Docs point at the cfg instead of restating its values. Tests and checks keep their own copy on purpose: it is the independent expectation the cfg is checked against.
- When a cfg value changes, grep `README.md`, `docs/`, `tests/` and `scripts/` for the old value and update every hit.

## Python

- Type hints and docstrings on public functions; Isaac Lab's config-class style.
- A module is created together with its code. The spec of an unbuilt feature lives in its issue.

## Verification

Much of this code is simulation and RL, where unit tests don't cover everything. Use checks like these as success criteria:
- The environment launches and steps without errors (short headless run with few envs).
- Observation/action shapes and value ranges match expectations.
- Physics sanity: the stem sags under gravity, springs back after release, and deflects when pushed.
- Reward terms and terminations fire in the cases they should (e.g. curvature limit exceeded).

Checks first: the expected values of `check_*.py`, sweeps and unit tests come from decisions (cfg tags, daily notes, write-ups), never from observed results, and are set before the code changes. A criterion is loosened only by the user's decision.

## Training runs

- Log the git commit hash with every training run.
