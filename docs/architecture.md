# Architecture

How the project is put together and where each future piece goes. Open items live in `docs/TODO.md`.

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
`scripts/` holds sanity checks and small tools. `tests/` holds unit tests that don't need the simulator.
`tools/autotrain/` holds the unattended training loop (driver, launch scripts, agent prompt and permissions);
its journal goes to `docs/experiments/`.

## Stem models

Stem models are selectable by name, like the end-effectors: `cable` (Newton cable) and `chain` (rigid-segment
articulation in PhysX, sharing one solver with the robot). The supervisors chose `chain` for the training
(2026-10-06); `cable` is kept for comparison.

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
  with joint armature 1e-4 kg m^2. No stretch or shear by construction.
- **Collision surface:** a smooth tube of the stem radius: each capsule's round ends are centred on the joints, so
  neighbours share a sphere there (capsules that only touch leave a groove at every joint, where the fork's slot
  edges caught the stem).
- **USD:** written from the yaml at spawn time into `assets/stem/chain/build/` (gitignored), geometry only.
  Stiffness, damping, armature and masses are runtime properties, randomizable per env without a new USD.
- **Armature:** PhysX solves the joint springs iteratively; without armature the 1 g segments on stiff springs
  give modes up to 3.6 kHz that 8 iterations do not resolve (stem 16 x too soft). Armature removes these modes:
  static shape exact, first mode -1 %, modes 2 / 3 -26 % / -64 %.
- **Solver:** PhysX TGS, 2 ms step, 8 position / 1 velocity iterations (`chain.yaml`).
- **Robot:** the FR3 already runs in PhysX (`fr3_cfg`), so robot and stem share one solver; contact forces via
  Isaac Lab's `ContactSensor` (checked in `scripts/check_contact.py`: within 1-5 % of the force from the stem's
  stiffness). Joint reaction forces (axial force, twist moment per joint) come from `chain.joint_wrenches(stem)`
  (PhysX `get_link_incoming_joint_force`; checked: equal to the weight above each joint).

## Physics (stem model `cable`, Newton cable)


- **Stem:** Isaac Lab `CableObject`, a chain of capsule segments joined by cable joints (stretch, shear,
  bend, twist), solved by Newton's VBD solver. Parameters in `assets/stem/stem.yaml` + `assets/stem/cable/cable.yaml`.
- **Robot:** FR3 + fork (end-effector `fork_v2`) in MuJoCo-Warp (Newton).
- **Coupling:** `CouplerProxyCfg`. The fork appears as a proxy collider in the VBD solve.
  Template: IsaacLab `isaaclab_tasks/core/lift/config/franka_soft/franka_cable_env_cfg.py`.
- The cable is Newton-only, so the whole scene runs on Newton. This is a first version: the stem model may
  change after discussion with the supervisors / the deformables expert.

## Data flow per step (stage 1)

1. **Action** (6-D): tool-tip translation and rotation step in the robot base frame, integrated into a target pose
   (speed caps 10 cm/s and 45 deg/s, acceleration limits 0.08 m/s^2 and 20 deg/s^2 per policy step) and executed by
   Franka's Cartesian impedance law with apparent-mass damping (`stem_manip.utils.impedance`, action term
   `ToolTipImpedanceAction`); at every physics step the target is kept within 4 mm / 3 deg of the tool
   (`substep_command`: bounds the contact force to 4 N). The arm is torque controlled, as on the real FR3. Details:
   `docs/overleaf_folder/open_questions/action_limits_problem.tex`.
2. **Simulation:** PhysX steps robot and stem (`chain`) together in one solver.
3. **Observation** (stem state read from the simulation; on the real robot it will come from cameras):
   57 values, robot base frame, raw meters: tool_tip position and orientation (2 rotation-matrix columns), applied
   step and target offset (21); stem base (3); 5 points along the stem at 0.08 ... 0.40 m, the last one the tip,
   for the previous and the current policy step (30); target position of the tip (3). The stem-state terms are in
   `tasks/push_position/mdp/stem_state.py`. No contact forces. To be reduced to what the perception can deliver.
4. **Reward / termination** (`mdp/rewards.py`, `mdp/terminations.py`, weights and thresholds in the env cfg):
   distance of the stem point to the target (coarse / fine tanh, the fine one an ellipsoid; height term at weight 0), fork-to-stem approach,
   curvature and contact-force penalties, action rate, early-termination penalty; terminate on time out, curvature
   limit, contact-force limit (both from `damage` in `stem.yaml`) and joint margin, no tool-tip bound. Contact forces
   (sensor `stem_contact` on the stem segments, filtered to the robot) and joint forces are used here, not in the
   observation.

The **stem point of interest** is given by a segment index + an offset along that segment (task cfg, `CommandsCfg`):
the tip. Its **target** is sampled once per episode around the tip's rest position (`stem_manip.utils.stem_target`):
3-10 cm sideways in any direction, height on the "bowl" the tip reaches when pushed (drop 0.6 r^2 / s) or below it,
down to the deepest stem shape within 0.8 x the curvature limit (C / S bends, which need a moment from the fork's slot).

## Stem interface (keeps the stem model swappable)

The stem model may change (the `chain` in PhysX is used, the Newton cable is kept; another
backend or a Cosserat co-simulation later). To keep that change local, **the rest of the code reads the stem only through
one accessor**:

```
stem_segment_poses(env) -> (num_envs, num_segments, 7)   # position + quaternion (x, y, z, w), world frame
```

- MDP terms (observations, rewards, terminations, commands) and `stem_manip.utils.stem_geometry` use only
  this accessor (plus segment rest lengths). They never call `CableObject` or any backend API directly.
- Switching the stem model then means changing the model name (see Stem models); the accessor is implemented
  per model. Task logic, rewards and training setup stay unchanged.
- Writing stem state (resets, randomization in `mdp/events.py`) is inherently model-specific; keep it in
  one place next to the accessor as well.
- Each stem model provides its asset cfg, physics cfg and base clamping (implemented), and the accessor and
  writing its state (added with the `chain` model, when both implementations exist).

## No-damage constraint

A stem can be damaged in several ways. Each needs its own measure:

| Damage mode | Cause | Measure | Available now? |
|---|---|---|---|
| Bending (kink / break) | pushing sideways too far | bending curvature per joint | yes, from segment poses |
| Torsion | twisting the stem | twist rate per joint | yes, from segment poses |
| Tearing / pulling out | fork drags along the stem with high friction, pulls the stem axially | tensile stress per joint = E · axial strain | yes, from segment poses (limited by float32 positions, see TODO) |
| Crushing / abrasion | high contact force or sliding under friction at the fork | contact normal / friction force | no: contact forces not readable in coupled scenes (see TODO open questions) |

Per joint i, with relative rotation q_rel = q_i^-1 * q_(i+1) and dual length L_dual = 0.5 * (L_i + L_(i+1)):
- q_rel is split into a **bending** part (rotation about an axis perpendicular to the stem tangent) and a
  **twist** part (rotation about the tangent), a swing-twist decomposition. The total angle of q_rel mixes both.
- bending curvature kappa_i = bend_angle_i / L_dual; twist rate tau_i = twist_angle_i / L_dual.
- axial strain eps_i = (gap between the end of segment i and the start of segment i+1, along the tangent) / L_dual.
  Pure bending gives zero (the distance between segment centres would cut the corner and report a false
  compression of about -angle²/8). Tensile stress sigma_i = E · eps_i, limit on tension only.

All measures are implemented once in `stem_manip.utils.stem_geometry` and used by rewards (penalties) and
terminations (hard limits from `damage` in `assets/stem/stem.yaml`). `stem_geometry` is geometry only;
material properties (stress = E · strain) enter in the MDP terms. Stage 1 enforces the bending limit and,
once a threshold is chosen, the tensile-stress limit; the twist limit follows when a realistic value is known.

## Where future work goes

| Item | Location |
|---|---|
| Stage 1: position control | `src/stem_manip/tasks/push_position/` |
| Stage 2: pose control | `src/stem_manip/tasks/push_pose/` |
| Domain randomization of stem parameters | `tasks/<task>/mdp/events.py`, ranges in the task's env cfg |
| Model-based baseline | `src/stem_manip/baselines/` |
| Another end-effector | `assets/fr3/end_effectors/<ee>/` (see `assets/fr3/README.md`) |
| Perception: camera-based stem state (with Alessio Caporali) | to decide; starts in parallel with M5 (`docs/TODO.md`) |
| Real robot deployment | later (stage 3) |
| Distillation into an end-to-end vision policy | stretch goal |

## Open modelling questions

Listed under "Open questions" in `docs/TODO.md`. The main ones for the cable model:
- the base is clamped by making the root segment a kinematic body in the Newton model (`fix_stem_base`), since
  Isaac Lab only offers pins (ball joints);
- no damping parameter is exposed in Isaac Lab; we author Newton's rod damping attributes through `StemMaterialCfg`;
- the VBD solver does not converge in bending with physical stretch and shear stiffness; both are softened in
  `cable/cable.yaml` (stretch 0.01 x, shear 0.001 x the bend modulus), with solver settings in its `solver` section;
- per-env randomization: candidate path via Newton's per-joint `joint_target_ke` / `joint_target_kd`, unverified;
- Newton contact sensors are not supported in coupled scenes.
