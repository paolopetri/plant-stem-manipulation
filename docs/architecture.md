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
- **Robot:** FR3 + fork in MuJoCo-Warp (Newton).
- **Coupling:** `CouplerProxyCfg`. The fork appears as a proxy collider in the VBD solve.
  Template: IsaacLab `isaaclab_tasks/core/lift/config/franka_soft/franka_cable_env_cfg.py`.
- The cable is Newton-only, so the whole scene runs on Newton. This is a first version: the stem model may
  change after discussion with the supervisors / the deformables expert.

## Data flow per step (stage 1)

1. **Action:** relative end-effector position, turned into joint targets by differential IK on the body
   `fork` with `tool_tip_offset("fork")`.
2. **Simulation:** Newton steps robot and stem together (proxy coupling, a few substeps).
3. **Observation** (privileged state): tool_tip pose, stem segment poses, position of the stem point of
   interest, target position, last action.
4. **Reward / termination:** distance of the stem point to the target, curvature penalty, action
   penalties; terminate on curvature limit, time out, out of bounds.

The **stem point of interest** is given by a segment index + an offset along that segment (task cfg).

## No-damage constraint

Curvature per cable joint from adjacent segment orientations:
kappa_i = angle(q_i^-1 * q_(i+1)) / L_dual, with L_dual = 0.5 * (L_i + L_(i+1)).
Implemented once in `stem_manip.utils.stem_geometry`, used by the reward (penalty) and the termination
(hard limit `damage.max_curvature` in `assets/stem/stem.yaml`).

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
- the base can only be pinned (ball joint), not clamped;
- no damping parameter is exposed;
- per-env randomization of the material is unclear;
- Newton contact sensors are not supported in coupled scenes.
