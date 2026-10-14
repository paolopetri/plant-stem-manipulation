# Architecture

How the project is put together and why. Values live in the cfg files this page points to; open work is tracked in
GitHub issues.

## Layers

```
assets/                     source data + build pipelines (URDF, meshes, yaml, generated USD)
   │
src/stem_manip/assets/      Isaac Lab configs that point to that data: fr3_cfg(ee), stem_model(name).stem_cfg()
   │
src/stem_manip/tasks/<task> one package per task: env cfg (scene + physics + MDP), mdp/ terms, agents/
   │   registered as a Gymnasium ID, found through the `isaaclab.tasks` entry point in pyproject.toml
   │
uv run isaaclab train/play --task <ID>      Isaac Lab's own CLI, nothing custom
```

The repo is an installable uv package (`stem_manip`), following Isaac Lab 3's external-project template.
`src/stem_manip/utils/` holds simulator-free helpers (unit-tested in `tests/`).
`scripts/` holds sanity checks and small tools. `tests/` holds unit tests that don't need the simulator.
`tools/autotrain/` holds the unattended training loop (driver, launch scripts, agent prompt and permissions);
its journal goes to `docs/experiments/`.

## Stem models

Stem models are selectable by name, like the end-effectors: `cable` (Newton cable) and `chain` (rigid-segment
articulation in PhysX, sharing one solver with the robot). The supervisors chose `chain` for the training
(2026-10-06); `cable` is parked for a later comparison. Why: `docs/adr/0001-chain-stem-model-in-physx.md`.

```
assets/stem/stem.yaml                  the plant, shared by all models: geometry, material, damping, damage limits
assets/stem/<model>/<model>.yaml       model-specific: segment count, extra springs (cable), solver settings
src/stem_manip/assets/stem/__init__.py stem_params(model), stem_model(name), STEM_MODELS
src/stem_manip/assets/stem/<model>.py  the model (functions below)
```

Each model module provides the same functions:

| Function | Purpose |
|---|---|
| `stem_cfg()` | Isaac Lab asset cfg of the upright stem (set `prim_path` in the scene) |
| `physics_cfg()` | physics cfg of the scene (the engine follows from the model) |
| `fix_stem_base(stem)` | clamp the base after the simulation is built (chain: nothing to do) |
| `segment_poses(stem)` | (num_envs, num_segments, 7) segment poses, the stem interface below |
| `segment_masses(stem)`, `joint_gains(stem)` | read back masses and per-joint stiffness / damping (/ armature) |
| `write_kick(stem, angular_velocity)` | rigid rotation of the stem above the first joint (checks) |
| `segment_force_setter(stem)` | `set_force(segment, force)`: constant world-frame force on a segment (checks) |


- `stem_params(model)` merges the two yaml files (same sections; a key may be defined in only one file, so both
  models simulate the same plant).
- `stem_model(name)` imports the model module on first use; the physics engine of the scene follows from the
  model (`physics_cfg()`). Scripts select the model with `--stem_model`.
- Adding a model: a folder `assets/stem/<name>/` with `<name>.yaml` and a module `<name>.py` with the functions
  above.

## Physics (stem model `chain`, PhysX articulation)

- **Stem:** Isaac Lab `Articulation` of rigid capsule segments. Segment 0 is clamped by a fixed joint; every other
  joint only rotates (D6 joint with locked translations, a spherical joint with 3 DOFs in PhysX: `joint_<i>:0`
  twist, `:1`/`:2` bend). Springs and dampers are implicit actuators (E I / l, G J / l, damping time x stiffness)
  with joint armature (`chain.yaml`). No stretch or shear by construction.
- **Collision surface:** a smooth tube of the stem radius: each capsule's round ends are centred on the joints, so
  neighbours share a sphere there (capsules that only touch leave a groove at every joint, where the fork's slot
  edges caught the stem).
- **USD:** written from the yaml at spawn time into `assets/stem/chain/build/` (gitignored), geometry only.
  Stiffness, damping, armature and masses are runtime properties, randomizable per env without a new USD.
- **Armature:** PhysX solves the joint springs iteratively; without armature the light segments on stiff springs
  give high-frequency modes that the solver iterations do not resolve, and the stem comes out far too soft.
  Armature removes these modes: the static shape stays exact, the first mode barely changes, the higher modes slow
  down (measurements: `docs/notes/2026-10-05.md`, `docs/overleaf_folder/open_questions/stem_model_comparison.tex`).
- **Solver:** PhysX TGS; step and iterations in `chain.yaml`.
- **Robot:** the FR3 already runs in PhysX (`fr3_cfg`), so robot and stem share one solver; contact forces via
  Isaac Lab's `ContactSensor` (checked against the force from the stem's stiffness in `scripts/check_contact.py`).
  Joint reaction forces (axial force, twist moment per joint) come from `chain.joint_wrenches(stem)`
  (PhysX `get_link_incoming_joint_force`; checked: equal to the weight above each joint).

## Physics (stem model `cable`, Newton cable, parked)

- **Stem:** Isaac Lab `CableObject`, a chain of capsule segments joined by cable joints (stretch, shear,
  bend, twist), solved by Newton's VBD solver. Parameters in `assets/stem/stem.yaml` + `assets/stem/cable/cable.yaml`.
- **Robot:** FR3 + fork (end-effector `fork_v2`) in MuJoCo-Warp (Newton).
- **Coupling:** `CouplerProxyCfg`. The fork appears as a proxy collider in the VBD solve.
  Template: IsaacLab `isaaclab_tasks/core/lift/config/franka_soft/franka_cable_env_cfg.py`.
- The cable is Newton-only, so the whole scene runs on Newton. Its open modelling questions (base clamping,
  damping, solver convergence with physical stretch and shear stiffness, per-env randomization, contact sensors in
  coupled scenes) are in #99; the remaining cable work is parked in #96 (base placement), #97 (FR3 on Newton) and
  #98 (coupled scene).

## Data flow per step (stage 1)

Task `StemManip-Push-Position-FR3-v0`, cfg in `src/stem_manip/tasks/push_position/env_cfg.py`.

1. **Action:** a tool-tip translation and rotation step in the robot base frame, integrated into a target pose and
   executed by Franka's Cartesian impedance law with apparent-mass damping (`stem_manip.utils.impedance`, action term
   `ToolTipImpedanceAction`). Speed caps, acceleration limits and the target clamp at every physics step (which
   bounds the contact force) are fields of `ToolTipImpedanceActionCfg` (`mdp/actions_cfg.py`). The arm is torque
   controlled, as on the real FR3. Reasons for the limits: `docs/overleaf_folder/open_questions/action_limits_problem.tex`.
   Why this controller: `docs/adr/0003-cartesian-impedance-action.md`.
2. **Simulation:** PhysX steps robot and stem (`chain`) together in one solver.
3. **Observation** (stem state read from the simulation; on the real robot it will come from cameras): robot base
   frame, raw meters; the tool tip's pose, applied step and target offset, the stem base, points along the stem
   (the last one the tip) for the previous and the current policy step, and the tip's target. Terms, their order and
   the arc lengths: `ObservationsCfg` in the env cfg; the stem-state terms are in `mdp/stem_state.py`. No contact
   forces (`docs/adr/0002-contact-forces-not-observed.md`). To be reduced to what the perception can deliver.
4. **Reward / termination** (`mdp/rewards.py`, `mdp/terminations.py`; weights and thresholds in `RewardsCfg` /
   `TerminationsCfg` of the env cfg, damage limits from `damage` in `assets/stem/stem.yaml`): distance of the stem
   point to the target (coarse and fine tanh, the fine one an ellipsoid that weighs the height error more) and a
   height term, fork-to-stem approach, curvature and contact-force penalties, action rate, early-termination
   penalty; terminate on time out, curvature limit, contact-force limit and joint margin, no tool-tip bound. Contact
   forces (sensor `stem_contact` on the stem segments, filtered to the robot) are used here, not in the observation.

The **stem point of interest** is given by a segment index + an offset along that segment: the tip. Its **target**
is sampled once per episode around the tip's rest position: sideways in any direction, at the height the tip reaches
when pushed (the "bowl") or below it, down to the deepest stem shape within a fraction of the curvature limit (C / S
bends, which need a moment from the fork's slot). Ranges: `StemTipTargetCommandCfg` (`mdp/commands.py`); shape model:
`stem_manip.utils.stem_target`. At every reset the stem base is moved to a random place in the spawn area
(`STEM_SPAWN_X` / `STEM_SPAWN_Y` in the env cfg); the target is sampled around the new base.

## Stem interface (keeps the stem model swappable)

The stem model may change (the `chain` in PhysX is used, the Newton cable is parked; another backend or a Cosserat
co-simulation later). To keep that change local, **the MDP reads the stem only through one accessor**:

```
mdp/stem_state.py: stem_segment_poses(env, model) -> (num_envs, num_segments, 7)   # position + quaternion (x, y, z, w), world frame
```

- It calls the model module's `segment_poses(stem)` (see Stem models). MDP terms (observations, rewards,
  terminations, commands) and `stem_manip.utils.stem_geometry` use only this accessor (plus segment rest lengths),
  never a backend API directly.
- Switching the stem model then means changing the model name; task logic, rewards and training setup stay
  unchanged.
- Writing stem state (base clamping and the spawn-area reset in `mdp/events.py`) goes through the model module or
  the asset's generic Isaac Lab API, kept in that one file.

## No-damage constraint

A stem can be damaged in several ways. Each needs its own measure:

| Damage mode | Cause | Measure | Available now? |
|---|---|---|---|
| Bending (kink / break) | pushing sideways too far | bending curvature per joint | yes, from segment poses |
| Torsion | twisting the stem | twist rate per joint | yes, from segment poses |
| Tearing / pulling out | fork drags along the stem with high friction, pulls the stem axially | tensile stress per joint = axial force / cross-section | yes: `chain` from the joint reaction force (`chain.joint_wrenches`; the chain cannot stretch, so its strain is zero); `cable` from the axial strain (E · strain) |
| Crushing / abrasion | high contact force or sliding under friction at the fork | contact force of the robot on the stem | yes: `ContactSensor` on the stem segments, filtered to the robot |

Per joint i, with relative rotation q_rel = q_i^-1 * q_(i+1) and dual length L_dual = 0.5 * (L_i + L_(i+1)):
- q_rel is split into a **bending** part (rotation about an axis perpendicular to the stem tangent) and a
  **twist** part (rotation about the tangent), a swing-twist decomposition. The total angle of q_rel mixes both.
- bending curvature kappa_i = bend_angle_i / L_dual; twist rate tau_i = twist_angle_i / L_dual.
- axial strain eps_i = (gap between the end of segment i and the start of segment i+1, along the tangent) / L_dual,
  for models that stretch (`cable`). Pure bending gives zero (the distance between segment centres would cut the
  corner and report a false compression of about -angle²/8). There, tensile stress sigma_i = E · eps_i; for the
  `chain` it is the axial joint force / A. Limit on tension only.

The geometric measures are implemented once in `stem_manip.utils.stem_geometry` (geometry only; material properties
enter in the MDP terms), the joint forces in the model module (`chain.joint_wrenches`). Rewards (penalties) and
terminations (hard limits from `damage` in `assets/stem/stem.yaml`) use them. Stage 1 enforces the bending and the
contact-force limits; the tension and twist limits follow when realistic values are known.

## Where future work goes

Open work is tracked in GitHub issues (`gh issue list`); this table says where its code goes.

| Item | Location |
|---|---|
| Stage 1: position control | `src/stem_manip/tasks/push_position/` |
| Stage 2: pose control (#103) | a new task package under `src/stem_manip/tasks/` |
| Domain randomization of stem parameters | `tasks/<task>/mdp/events.py`, ranges in the task's env cfg |
| Another end-effector | `assets/fr3/end_effectors/<ee>/` (see `assets/fr3/README.md`) |
| Perception: camera-based stem state (with Alessio Caporali) | to decide |
| Real robot deployment (stage 3) | to decide |
| Distillation into an end-to-end vision policy (stretch goal) | to decide |

## Open modelling questions

The cable path's open questions are in #99 (parked, with #96-#98); the open questions of the current path are
issues as well.
