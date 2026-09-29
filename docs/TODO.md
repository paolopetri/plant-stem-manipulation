# TODO

Single source for what comes next. When an item is done, delete it here and log it in today's `docs/notes/` file.

## Next
- Convert URDF → USD (`assets/fr3_fork/convert_to_usd.sh`). Check the converter flags for the installed Isaac Lab version and make sure fixed joints are not merged (`fork` and `tool_tip` must exist as bodies).
- Load `FR3_FORK_CFG` in Isaac Lab: short headless run, check body names and the initial pose.

## Blocked / waiting
- Weigh the printed fork → set `inertial.measured_mass` in `fork.yaml`.

## Repo
- Add a root `.gitignore` (logs, checkpoints, videos, `wandb/`). CLAUDE.md says they are ignored, but they aren't yet.
- Add `docs/project-proposal.md` (referenced in CLAUDE.md).
- Fill in "Repo layout" and "Commands" in CLAUDE.md.
