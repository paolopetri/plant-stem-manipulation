# TODO

Single source for what comes next. When an item is done, delete it here and log it in today's `docs/notes/` file.

## Next
- Convert URDF → USD (`assets/fr3/convert_to_usd.sh fork`). Check the converter flags for the installed Isaac Lab version and make sure fixed joints are not merged (`fork` and `tool_tip` must exist as bodies).
  - Fixed joints: `convert_urdf.py` defaults to `--merge_joints` off and our script doesn't set it, so `fork` should stay a separate body. (`UrdfConverterCfg` in Python defaults to merging, so it matters if we ever convert from code.)
  - `tool_tip` is probably not a body: `urdf_usd_converter` 0.3.2 turns links with no mass, no geometry and a fixed joint into plain frames ("ghost links", `link_hierarchy.py`). Then `EE_BODY = "tool_tip"` in `fr3_cfg.py` finds nothing. Check the body names after conversion; if missing, use `fork` + fixed offset or a `FrameTransformer`.
- Load `fr3_cfg("fork")` (`assets/fr3/fr3_cfg.py`) in Isaac Lab: short headless run, check body names and the initial pose.

## Later
- When a second arm is added: move `convert_to_usd.sh` and `view_urdf.py` to `assets/tools/` with a URDF path as argument, and generalize the base handling in `build_asset.py` (base URDF, description package, mount link).

## Blocked / waiting
- Weigh the printed fork → set `inertial.measured_mass` in `fork.yaml`.

## Repo
- Add a root `.gitignore` (logs, checkpoints, videos, `wandb/`). CLAUDE.md says they are ignored, but they aren't yet.
- Add `docs/project-proposal.md` (referenced in CLAUDE.md).
- Fill in "Repo layout" and "Commands" in CLAUDE.md.
