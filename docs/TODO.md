# TODO

Single source for what comes next. When an item is done, delete it here and log it in today's `docs/notes/` file.
Roadmap to the first training run (stage 1, position control). Milestones in order; each is one or more branches.
Where things go: `docs/architecture.md`. Spec of each skeleton file: its module docstring.

## M0 Repo skeleton (`refactor/project-skeleton`)
- Verify the headless FR3 check with the new import path (`stem_manip.assets.fr3`, PhysX): bodies `fr3_link0`…`fr3_link7` + `fork`, start pose held. Blocked on first-launch EULA acceptance in the new `.venv`.

## M1 Stem model v1 (Newton cable)
- Fill `assets/stem/stem.yaml` with nominal values (length, segments, diameter, density, moduli, curvature limit).
- `stem_cfg()` in `src/stem_manip/assets/stem.py` builds a `CableObjectCfg` from the yaml.
- Hold the stem base fixed. Try pinning the first two control points (a single pin is only a ball joint); fall back to other workarounds, document the choice.
- `stem_manip.utils.stem_geometry`: per-joint curvature and point-of-interest pose; `tests/test_stem_geometry.py` passes.
- `scripts/check_stem.py` passes: base fixed, stem stands and sags plausibly, springs back, deflects when pushed.

## M2 FR3 on Newton
- `fr3_cfg("fork")` loads and holds its pose under Newton / MuJoCo-Warp (currently PhysX-only schemas).
- Decide the gravity question (see Open questions) and apply it.
- Test the `tool_tip` offset in practice: IK with body `fork` + `tool_tip_offset("fork")` moves the `tool_tip` to the target, and a `FrameTransformer` with the same offset reports the `tool_tip` pose. So far only the offset values and a manual combination with the fork pose are checked.
- `scripts/check_fr3_newton.py` covers all of the above.

## M3 Coupled scene
- Robot + stem + ground with `CouplerProxyCfg` (robot in MuJoCo-Warp, stem in VBD, fork as proxy collider).
- `scripts/check_scene.py`: scripted push, stem deflects and springs back, no instabilities; report max curvature.

## M4 Stage-1 environment (`tasks/push_position`)
- Commands (target position of the stem point), observations, relative EE-position action.
- Rewards: distance to target, curvature penalty, action penalties. Terminations: curvature limit, time out, out of bounds. Reset events.
- Register `StemManip-Push-Position-FR3-v0`.
- Zero/random agent runs headless with few envs; observation/action shapes and ranges as expected; each termination shown to fire.

## M5 First training run
- `rsl_rl_ppo_cfg.py`; short headless training run; mean reward increases.
- Check that the run logs our repo's git commit.
- Tag `v0.1-position-control`; add train/play commands to README and CLAUDE.md.

## Later
- Domain randomization of stem parameters (stiffness, length, diameter, damping), after the per-env randomization question is answered.
- Stage 2: full pose control (`tasks/push_pose`).
- Model-based baseline (nominal vs. oracle parameters) in `src/stem_manip/baselines/`.
- Real-robot validation on an artificial plant; distillation into a vision-based policy.
- When a second arm is added: move `convert_to_usd.sh` and `view_urdf.py` to `assets/tools/` with a URDF path as argument, and generalize the base handling in `build_asset.py` (base URDF, description package, mount link).

## Open questions (supervisors / deformables expert)
- Stem model: is the Newton cable (discrete elastic rod, VBD) good enough, or a self-built rigid-segment articulation / Cosserat co-simulation?
- Cable base can only be pinned (ball joint), not clamped. Recommended way to clamp it?
- Cable damping is not exposed in Isaac Lab. How to set / randomize it?
- Per-env randomization of the cable material (authored in USD at spawn): supported path?
- Contact forces between fork and stem: Newton contact sensors are not supported in coupled scenes. Alternative?
- Gravity compensation: with `disable_gravity=False` and stiffness 400, the arm sags ~0.05 rad at joints 2 and 4 in the start pose. Disable gravity on the robot (as Isaac Lab's Franka high-PD config does), add gravity compensation, or raise the gains?
  - Leaning: do it like Isaac Lab's Franka. `FRANKA_PANDA_HIGH_PD_CFG` (`isaaclab_assets/robots/franka.py`) uses the same gains (400/80) plus `disable_gravity=True`, "useful for task-space control using differential IK". Isaac Lab's OSC how-to and gear-assembly deployment docs do the same ("Robot is mounted, no gravity"). The real Franka controller also compensates gravity itself (from memory, check in the libfranka docs).
  - Side effect: the end-effector then also has no gravity in sim. Negligible for the ~50 g fork, but worth knowing.

## Blocked / waiting
- Weigh the printed fork → set `inertial.measured_mass` in `fork.yaml`.

## Repo
- Add `docs/project-proposal.md` (referenced in CLAUDE.md).
