# Architecture

How the project is put together and where each future piece goes. Open items live in `docs/TODO.md`.

## Layers

```
assets/                     source data + build pipelines (URDF, meshes, yaml, generated USD)
   │
src/stem_manip/assets/      Isaac Lab configs that point to that data: fr3_cfg(ee), stem_cfg()
   │
src/stem_manip/tasks/<task> one package per task: env cfg (scene + physics + MDP), mdp/ terms, agents/
   │   registered as a Gymnasium ID, found through the `isaaclab.tasks` entry point in pyproject.toml
   │
uv run isaaclab train/play --task <ID>      Isaac Lab's own CLI, nothing custom
```

The repo is an installable uv package (`stem_manip`), following Isaac Lab 3's external-project template.
`scripts/` holds sanity checks and small tools. `tests/` holds unit tests that don't need the simulator.

## Physics (stem model v1)

- **Stem:** Isaac Lab `CableObject`, a chain of capsule segments joined by cable joints (stretch, shear,
  bend, twist), solved by Newton's VBD solver. Parameters in `assets/stem/stem.yaml`.
- **Robot:** FR3 + fork (end-effector `fork_v2`) in MuJoCo-Warp (Newton).
- **Coupling:** `CouplerProxyCfg`. The fork appears as a proxy collider in the VBD solve.
  Template: IsaacLab `isaaclab_tasks/core/lift/config/franka_soft/franka_cable_env_cfg.py`.
- The cable is Newton-only, so the whole scene runs on Newton. This is a first version: the stem model may
  change after discussion with the supervisors / the deformables expert.

## Data flow per step (stage 1)

1. **Action:** relative end-effector position, turned into joint targets by differential IK on the body
   `fork_v2` with `tool_tip_offset("fork_v2")`.
2. **Simulation:** Newton steps robot and stem together (proxy coupling, a few substeps).
3. **Observation** (privileged state): tool_tip pose, stem segment poses, position of the stem point of
   interest, target position, last action.
4. **Reward / termination:** distance of the stem point to the target, curvature penalty, action
   penalties; terminate on curvature limit, time out, out of bounds.

The **stem point of interest** is given by a segment index + an offset along that segment (task cfg).

## Stem interface (keeps the stem model swappable)

The stem model may change (Newton cable now; possibly a rigid-segment articulation, another backend or a
Cosserat co-simulation later). To keep that change local, **the rest of the code reads the stem only through
one accessor**:

```
stem_segment_poses(env) -> (num_envs, num_segments, 7)   # position + quaternion (x, y, z, w), world frame
```

- MDP terms (observations, rewards, terminations, commands) and `stem_manip.utils.stem_geometry` use only
  this accessor (plus segment rest lengths). They never call `CableObject` or any backend API directly.
- Switching the stem model then means changing two things: the asset cfg (`stem_manip.assets.stem`) and the
  accessor implementation. Task logic, rewards and training setup stay unchanged.
- Writing stem state (resets, randomization in `mdp/events.py`) is inherently model-specific; keep it in
  one place next to the accessor as well.

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
| Real robot, vision distillation | out of scope for now |

## Open modelling questions

Listed under "Open questions" in `docs/TODO.md`. The main ones for the cable model:
- the base is clamped by making the root segment a kinematic body in the Newton model (`fix_stem_base`), since
  Isaac Lab only offers pins (ball joints);
- no damping parameter is exposed in Isaac Lab; we author Newton's rod damping attributes through `StemMaterialCfg`;
- the VBD solver does not converge in bending with physical stretch and shear stiffness; both are softened in
  `stem.yaml` (stretch 0.01 x, shear 0.001 x the bend modulus), with solver settings in its `solver` section;
- per-env randomization: candidate path via Newton's per-joint `joint_target_ke` / `joint_target_kd`, unverified;
- Newton contact sensors are not supported in coupled scenes.
