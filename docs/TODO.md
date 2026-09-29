# TODO

Single source for what comes next. When an item is done, delete it here and log it in today's `docs/notes/` file.

## Next
- First environment (stage 1, position control of the stem point): when setting it up, test the `tool_tip` offset in practice. The IK action with body `fork` + `tool_tip_offset("fork")` must move the `tool_tip` to the target, and a `FrameTransformer` with the same offset must report the `tool_tip` pose. So far only the offset values and a manual combination with the fork pose are checked.

## Later
- When a second arm is added: move `convert_to_usd.sh` and `view_urdf.py` to `assets/tools/` with a URDF path as argument, and generalize the base handling in `build_asset.py` (base URDF, description package, mount link).

## Open questions (supervisors)
- Gravity compensation: with `disable_gravity=False` and stiffness 400, the arm sags ~0.05 rad at joints 2 and 4 in the start pose. Disable gravity on the robot (as Isaac Lab's Franka high-PD config does), add gravity compensation, or raise the gains?
  - Leaning: do it like Isaac Lab's Franka. `FRANKA_PANDA_HIGH_PD_CFG` (`isaaclab_assets/robots/franka.py`) uses the same gains (400/80) plus `disable_gravity=True`, "useful for task-space control using differential IK". Isaac Lab's OSC how-to and gear-assembly deployment docs do the same ("Robot is mounted, no gravity"). The real Franka controller also compensates gravity itself (from memory, check in the libfranka docs).
  - Side effect: the end-effector then also has no gravity in sim. Negligible for the ~50 g fork, but worth knowing.

## Blocked / waiting
- Weigh the printed fork → set `inertial.measured_mass` in `fork.yaml`.

## Repo
- Add a root `.gitignore` (logs, checkpoints, videos, `wandb/`). CLAUDE.md says they are ignored, but they aren't yet.
- Add `docs/project-proposal.md` (referenced in CLAUDE.md).
- Fill in "Repo layout" and "Commands" in CLAUDE.md.
