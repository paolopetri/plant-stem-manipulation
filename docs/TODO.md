# TODO

Single source for what comes next. When an item is done, delete it here and log it in today's `docs/notes/` file.
Roadmap to the first training run (stage 1, position control). Milestones in order; each is one or more branches.
Where things go: `docs/architecture.md`. Spec of each skeleton file: its module docstring.

## M1 Stem model v1 (Newton cable)
Done (2026-10-01). The stem values stay placeholders: the policy should work for any plant, and the values will be randomized (see Later).

## M1b Stem model v2: rigid-segment chain in PhysX (prototype), modular stem models
Why: with the Newton cable, realistic stretch/shear stiffness, correct bending and speed exclude each other (open questions, stem-model entry), and robot and stem need two coupled solvers (open questions, contact handling). A chain of rigid segments connected by joints that only bend and twist cannot stretch or shear by construction, and in PhysX it shares one solver with the robot (contact forces readable, no coupling). Decision 2026-10-01: prototype it now, keep the Newton cable, make the stem model selectable like the end-effectors, and decide later (with the supervisors) which model M2-M4 use.
- Exact reference for the chain in `stem_manip.utils.stem_reference` (small-deflection model with gravity and armature: mode frequencies, push deflection) with a unit test, used by `check_stem.py`. The comparison numbers of 2026-10-05 come from a scratch version of it.
- First contact test in PhysX: FR3 with `fork_v2` (already runs in PhysX, `check_fr3.py`) pushes the `chain` stem. Check: stable contact between the fork and the light segments (mass ratio about 1 g vs. kilograms), no tunnelling, contact force readable with Isaac Lab's `ContactSensor`.
- Compare `cable` and `chain` in one table (accuracy in the tests, time per step with many envs, contact behaviour, effort) and decide with the supervisors which model M2-M4 use. Both stay in the repo.

## M2 FR3 on Newton
Path for the `cable` model. If `chain` in PhysX is chosen, M2 becomes "FR3 + stem in one PhysX scene" and M3 (coupling) is not needed.
- `fr3_cfg("fork_v2")` loads and holds its pose under Newton / MuJoCo-Warp (currently PhysX-only schemas).
- Decide the gravity question (see Open questions) and apply it.
- Test the `tool_tip` offset in practice: IK with body `fork_v2` + `tool_tip_offset("fork_v2")` moves the `tool_tip` to the target, and a `FrameTransformer` with the same offset reports the `tool_tip` pose. So far only the offset values and a manual combination with the fork pose are checked.
- `scripts/check_fr3_newton.py` covers all of the above.

## M3 Coupled scene
- Contact settings for the fork-stem contact: `NewtonShapeCfg` (ke, kd, mu); Isaac Lab's cable task uses ke=2.5e3, kd=100, mu=10.
- Robot + stem + ground with `CouplerProxyCfg` (robot in MuJoCo-Warp, stem in VBD, fork as proxy collider).
- Lay out the envs with `env_spacing=0` (all envs stacked at the world origin; Newton keeps them from colliding). Check that this still holds with the robot and the coupled solvers, and that observations do not depend on it.
- Re-run the stem accuracy tests (`scripts/sweep_stem_solver.py`) with the solver settings of the coupled scene; the values in `cable/cable.yaml` (`solver`) were checked for the stem alone.
- `scripts/check_scene.py`: scripted push, stem deflects and springs back, no instabilities; report max curvature.

## M4 Stage-1 environment (`tasks/push_position`)
- Commands (target position of the stem point), observations, relative EE-position action.
- Call `fix_stem_base` once at env start (startup event), and check that it survives resets.
- Rewards: distance to target, curvature penalty, action penalties. Terminations: curvature limit, time out, out of bounds. Reset events.
- Tensile-stress limit (tension only), compared with `damage.max_tensile_stress`. `chain`: stress = axial joint force / A, read exactly from `stem.root_view.get_link_incoming_joint_force()` (joint x axis, + = compression; checked against the weight above each joint and a 1 N pull, 2026-10-05). `cable`: stress = `stretch_modulus` · `joint_axial_strain`; choose the threshold and the shape (free zone below it, penalty / termination above). The strain is not reliable yet: at cost 80 the upright stem at rest is compressed 2.3 x more than by hand (base joint -1.72e-3 vs. -7.46e-4; 40 iterations: -8.2e-4), see the `[INFO] axial strain` lines of `check_stem.py`. Decide the solver cost or another tension measure before using it.
- Register `StemManip-Push-Position-FR3-v0`.
- Zero/random agent runs headless with few envs; observation/action shapes and ranges as expected; each termination shown to fire.

## M5 First training run
- `rsl_rl_ppo_cfg.py`; short headless training run; mean reward increases.
- Log our repo's git commit with every run: Isaac Lab's `train` stores the git state of the Isaac Lab and RSL-RL repos only (`runner.add_git_repo_to_log(__file__)` in `isaaclab_rl/entrypoints/backends/train_rsl_rl.py`). Option: a commit-hash field in our env cfg, which ends up in the run's `params/env.yaml`.
- Tag `v0.1-position-control`; add train/play commands to README and CLAUDE.md.

## Later
- Domain randomization of stem parameters (stiffness, length, diameter, damping), after the per-env randomization question is answered. Choose the ranges from plausible orders of magnitude for plant stems (literature, a cantilever test on the artificial plant as one data point) rather than from one reference plant; the placeholder values in `stem.yaml` only need to lie inside them. Re-check the solver settings over the whole range (the stretch/shear-to-bend ratio that converges depends on segment length / diameter, and soft shear adds about 10 % deflection, which the range covers).
- Stem damping: with `damping_time` 3.2 ms (damping ratio about 0.06) the stem swings back and forth several times after a kick, more like a bamboo stick than a living plant stem, which is likely much more damped. Fine for now; revisit with real stem values (measure the decay on the artificial plant) and in the randomization range.
- Stage 2: full pose control (`tasks/push_pose`).
- Model-based baseline (nominal vs. oracle parameters) in `src/stem_manip/baselines/`.
- Real-robot validation on an artificial plant; distillation into a vision-based policy.
- When a second arm is added: move `convert_to_usd.sh` and `view_urdf.py` to `assets/tools/` with a URDF path as argument, and generalize the base handling in `build_asset.py` (base URDF, description package, mount link).

## Open questions (supervisors / deformables expert)
- Could the robot also run in VBD (Newton), avoiding the coupling? VBD supports rigid bodies and revolute joints with drives, but treats joints as stiff springs; Isaac Lab's cable task couples MuJoCo-Warp + VBD instead. Untested. The stem results above make it unlikely to work well: stiff joints next to soft ones are exactly what VBD converges slowly on (expectation, not measured).
- **Stem model: realistic stretch/shear stiffness, correct bending and affordable simulation exclude each other.** To discuss with the supervisors / the deformables expert. Status: option (d) prototyped as stem model `chain` (2026-10-05, comparison in that day's note); the cable uses option (a) (stretch 0.01, shear 0.001, damping on bend and twist only, 8 x 10), marked "current" below.
  - *The stem model.* The stem is a chain of 20 rigid segments (Newton cable). Each joint has four springs: bend, twist, stretch (along the stem) and shear (neighbouring cross-sections sliding sideways). A real thin stem bends easily but practically does not stretch or shear, so physically the stretch and shear springs are far stiffer than the bending springs.
  - *How the solver works.* The VBD solver does not solve the equations of motion exactly. At every time step it starts from a guess and improves it a fixed number of times (substeps x iterations), moving one segment at a time while its neighbours are held fixed.
  - *Why unequal stiffness is a problem.* In the exact solution there is no conflict: the stiff springs decide what the stem cannot do (stretch, shear), the soft bending springs decide what it does within those rules. The problem is reaching that solution. The error the solver minimizes is a long narrow valley: steep walls (stretching and shearing cost a lot) and an almost flat floor (bending costs little). The solver drops down the walls in a few passes and then creeps along the floor. Stopped after a fixed number of passes, stretch and shear are satisfied but bending is not. Mechanically: to bend at one joint, everything beyond it must swing together; a single segment turning on its own moves its ends sideways against its fixed neighbours, which the stiff shear springs punish, so a bend is passed along the chain in tiny steps.
  - *What the error looks like.* Not noise, but a wrong bending stiffness, in either direction: too soft where the stem should hold itself up (it has not caught up with its own restoring force), too stiff where it should yield to a push (it has not moved far enough yet).
  - *It is a ratio, fixed by geometry.* What matters is stretch/shear stiffness relative to bending stiffness, roughly (shear modulus / bend modulus) x (segment length / diameter)^2 (own estimate, tested on one geometry: 0.4 m stem, 8 mm diameter, 20 mm segments). Scaling all moduli together or changing units changes nothing. With physical moduli the segments would have to be about 1 mm long (hundreds of segments).
  - *Measurements.* Stem alone, clamped base, 10 ms step, placeholder stem (0.4 m, 8 mm, 20 segments, bend modulus 5e8 Pa). Moduli are given as multiples of the bend modulus. Cost = substeps x iterations per step. Three tests: **sag** (stem clamped horizontally, tip sag under its own weight, divided by the hand value, so 1 is correct), **kick** (upright stem, tip kicked with 1 m/s, should swing and return upright), **push** (upright stem, 0.05 N sideways on the last segment; expected about 10.2 mm: hand value 9.85 mm without gravity plus about 4 % from the stem's own weight). Hand values: `stem_manip.utils.stem_reference`. Reproduce with `scripts/sweep_stem_solver.py` (sag, kick) and `scripts/check_stem.py` (push).

    Table 1: stretch and shear lowered together (damping on all four springs). Sag / hand value:

    | Stretch = shear | Cost 80 | Cost 160 | Cost 800 |
    |---|---|---|---|
    | 1 (physical order of magnitude) | not run | 23 (stem falls over in the kick test) | 15, still moving |
    | 0.1 | not run | 22 | 7 |
    | 0.01 | not run | 3.7 | 1.28 |
    | 0.001 | 1.30 to 1.44 | 1.06 | 1.003 |

    Table 2: which spring causes it (cost 80 = 8 substeps x 10 iterations, shear 0.001, damping on bend and twist only):

    | Stretch | Sag / hand value | Kick | Push deflection (expected about 10.2 mm) | Stretch under 1 N pull |
    |---|---|---|---|---|
    | 1 | 1.05 | blows up | not run | 0.016 mm |
    | 0.1 (used until the push test) | 1.012 | returns; 5.05 Hz; damping ratio 0.059 | 8.56 mm (16 % too stiff) | 0.16 mm |
    | 0.03 | 1.009 | returns; 4.75 Hz; 0.061 | 9.91 mm (3 % too stiff) | 0.5 mm |
    | 0.01 (current) | 1.008 | returns; 4.67 Hz; 0.063 | 10.27 mm (1 % too soft) | 1.6 mm |
    | 0.001 | not run | not run | 10.38 mm (2 % too soft) | 16 mm |

    Lowering only the shear modulus (row 1) already gives the correct sag, so shear is the main cause; but full stretch stiffness makes the kick test blow up, and gravity along the upright stem loads the stretch springs, which the push test shows and the sag test does not (without gravity, stretch 0.1 gives 10.06 mm in the push test). The 5.05 Hz of stretch 0.1 is the too-stiff stem; the softer rows give the frequency of the converged model.

    Table 3: can more solver work fix stretch 0.1 (shear 0.001)? Push deflection, expected about 10.2 mm:

    | Substeps x iterations | Cost | Push deflection |
    |---|---|---|
    | 8 x 10 | 80 | 8.56 mm |
    | 16 x 5 | 80 | 9.40 mm |
    | 8 x 15 | 120 | 9.11 mm |
    | 12 x 10 | 120 | 9.49 mm |
    | 8 x 20 | 160 | 9.39 mm |
    | 16 x 10 | 160 | 9.87 mm |
    | 8 x 40 | 320 | 9.86 mm |

    Table 4: damping on stretch and shear (tau x E A) makes cheap settings unstable (stretch 0.1, shear 0.001, kick test):

    | Damped springs | Cost 160 | Cost 80 | Cost 40 |
    |---|---|---|---|
    | all four | returns, damping ratio 0.05 | does not return (39 mm off) | blows up |
    | bend and twist only (current) | returns, 0.05 to 0.06 | returns, 0.05 to 0.07 | does not return (89 mm off) |

    Computing time with 1024 stems (stem only): 8.6 ms per 10 ms step at cost 80, 16.8 ms at cost 160.
  - *What softening costs.* Stretch: the stem elongates 0.16 mm per N of pull at 0.1 x, 0.5 mm at 0.03 x, 1.6 mm at 0.01 x (a real stem: practically nothing). Shear at 0.001 x: the stem bends about 10 % more than its bending stiffness alone allows (8 % for a force at the tip), which could be compensated by a 10 % higher bend modulus. Stretch and shear vibrations are undamped apart from the solver's numerical damping (ratio 0.01-0.02). The tensile-stress limit must use the stretch modulus actually simulated.
  - *Limits of these results.* One stem geometry with placeholder values, no contact, no robot. An accuracy claim only holds for the load cases tested (sag, kick, push); the ratios must be re-checked when stem parameters change or are randomized, and in the coupled scene with the fork.
  - *Options.*

    | | Change | Solver cost | Bending (sag / push) | Stretch under 1 N | Price |
    |---|---|---|---|---|---|
    | earlier | stretch 0.1, shear 0.001 | 80 | 1 % / 16 % too stiff | 0.16 mm | push check fails |
    | (a), current | stretch 0.01 | 80 | 1 % / 1 % | 1.6 mm | stem 10 x stretchier than with 0.1 |
    | (b) | stretch 0.03 | 80 | 1 % / 3 % too stiff | 0.5 mm | compromise between 0.1 and (a) |
    | (c) | stretch 0.1, 16 x 10 | 160 | 1 % / 3 % too stiff | 0.16 mm | half the simulation speed |
    | (d) | different stem model | unknown | exact by construction | none | build and validate a new model |

    (d) is a chain of rigid segments connected by joints that only bend and twist (an articulation with spring joints, stiffness E I / l per joint). It cannot stretch or shear by construction, which is the thin-rod assumption, so the stiff springs do not exist. It would run in the robot's solver (MuJoCo-Warp or PhysX), without coupling two solvers, with a clamped base and joint damping built in. A Cosserat-rod co-simulation (DeformX) was checked on 2026-09-29: it targets Isaac Sim 4.5/5.x and is not usable with Isaac Lab 3 without porting.
  - *Why not solve bending and shear separately?* Because they are not separate unknowns. The solver's unknowns are the position and orientation of every segment, and both springs depend on them: turning a segment changes the bend angle at its joints and at the same time moves its ends sideways (shear). A bend-only solve followed by a shear-only solve would undo each other and would have to alternate many times, with the same slow convergence. The stiff directions can be taken out of the problem in two ways, and both are a change of model or solver: describe the stem by its joint angles, so that stretch and shear cannot occur at all (option (d)); or solve all segments of the chain at once with a direct solver for rods, which handles any stiffness ratio (Newton's VBD updates one segment at a time and does not offer this). So the difficulty comes from the combination of this stem description (every segment free, held together by stiff springs) with this kind of solver; changing either one removes it, while softening only makes it mild enough.
  - *Questions.* Is softening stretch and shear the accepted practice for the Newton cable, and how soft is acceptable for a plant stem? Is there a solver setting or formulation that handles the stiff directions (compliant ALM, `rigid_compliant_alm=True`, made no difference)? Is model (d) the better choice for a stiff, thin stem, given that domain randomization of stiffness and diameter will move the ratio further?
- **Contact handling between fork and stem** (crux of the project, to discuss with the supervisors). The stem must be moved by contact forces (normal force and friction, with sliding), not by imposed displacements. Nothing is implemented yet; the plan so far (A) is copied from Isaac Lab's cable task. Options (read in the Isaac Lab / Newton source, none run in this project):

  | | How contact is handled | Force back on the robot | Contact force readable | Advantage | Drawback |
  |---|---|---|---|---|---|
  | A | proxy coupling (`CouplerProxyCfg`): robot in MuJoCo-Warp, stem in VBD, a copy of the fork placed in the VBD solve; the force on the copy is applied to the robot in the next step | yes, one step late | not with Isaac Lab's sensor | working Isaac Lab example | two solvers, delay, tuning (proxy mass, relaxation), inherits the cable's stiffness problem |
  | B | ADMM coupling (`CouplerAdmmCfg`): same two solvers, iterate to agree on the contact force within each step | yes, same step | not with Isaac Lab's sensor | no delay | more expensive; experimental in Newton |
  | C | one-way: only the stem is simulated; the fork follows the commanded path exactly (kinematic body) | no | yes, from the stem solver (to verify) | one solver, simple, fast | the stem cannot slow or deflect the robot; IK and joint limits handled separately |
  | D | one solver: stem as a rigid-segment chain next to the robot (PhysX, or MuJoCo-Warp) | yes, same step | yes (`ContactSensor`) | no coupling; also removes the stiffness problem | the stem model must be built and validated (M1b) |
  | E | quasi-static (Alessio Caporali's suggestion): for each fork position, compute the shape the stem settles into | not applicable | yes, part of the calculation | cheap, no time stepping | friction and sliding depend on history and are hard to represent; contact solver written by us |

  The deciding questions: does the force on the robot matter (a stiff position-controlled FR3 is hardly moved by forces below 1 N, which favours C), must the contact force be readable (crushing limit, rewards), and how realistic must friction and sliding be? Contact parameters (stiffness, damping, friction coefficient) are open for every option; Isaac Lab's cable task uses mu = 10, which suits gripping, not pushing. Current direction (2026-10-01): D, prototyped in M1b; the other options stay documented.
- Cable base (solved, confirmation wanted): Isaac Lab only offers pins (ball joints). We clamp the base by marking the root segment as a kinematic body in the Newton model (`fix_stem_base`); `check_stem.py` shows the base segment does not move at all, in several envs. Is there a cfg-level way, and does the flag survive env resets (to check in M4)?
- Cable damping (solved, confirmation wanted): Isaac Lab's `CableMaterialCfg` does not expose it. We author Newton's `newton:curves{Bend,Twist}Damping` on the material prim through a subclass (`StemMaterialCfg`); the values arrive in `model.joint_target_kd` and the measured damping ratio matches (`check_stem.py`). Stretch and shear are left undamped on purpose (see the stem-model entry). Is that the intended way?
- Per-env randomization of stiffness and damping: Newton stores rod stiffness/damping per joint in `model.joint_target_ke` / `joint_target_kd` (all envs in one array). The VBD solver caches them at construction; its documentation says to call `notify_model_changed(JOINT_DOF_PROPERTIES)` after editing (in Isaac Lab: `NewtonManager.add_model_change`, the route `fix_stem_base` already uses for body flags). Read in the source, not run. Randomizing length or diameter changes the geometry and the mass, which this path does not cover. Confirm with the expert.
- Contact forces between fork and stem: Isaac Lab's contact sensor is not supported with coupled solvers (contacts live in per-solver buffers, `isaaclab_contrib/coupling/coupler.py`). Candidate (unverified): the proxy-coupled solver's `get_proxy_contacts(source, destination)`. Needed only for force-based limits/rewards; stage 1 uses curvature.
- Joint armature for the `chain` (2026-10-05): PhysX solves the joint springs iteratively, and without armature the 1 g segments on E I / l springs come out 16 x too soft (8 iterations; 128 iterations still 4 x). Armature 1e-4 kg m² per joint coordinate fixes the static shape and the first mode (-1 %) but makes modes 2 and 3 26 % and 64 % too slow. Is that acceptable for slow pushing, or is a solver without this problem preferable (e.g. MuJoCo-Warp, which solves joint-space dynamics with a direct solver; untested)?
- Rigid segments and local loads (user remark, 2026-10-05): with rigid segments the fork acts on a rigid body part, so a twisting or crushing load at the contact is spread over the segment and the neighbouring joints, while a real stem deforms and can break locally. The policy could learn to twist or press a segment hard without being punished. Keep in mind when the damage limits are defined: limit the twist rate / joint twist torque (`max_twist_rate` is still null) and the contact force, not only the bending curvature.
- Damage criteria beyond bending: the measures exist (twist rate, axial strain; tearing is limited as a tensile stress, `damage.max_tensile_stress`), but the limits are not set. Realistic values for torsion and for tension (the fork dragging along the stem with friction)? Crushing / abrasion needs contact forces (see above). Literature values or measurements on the artificial plant?
- float32 world positions (largely answered): envs far from the world origin simulate a slightly different stem. With 1024 envs at 1 m spacing the push deflection is 10.1-10.3 mm near the origin but 9.45 mm 9.5-15.5 m away (7 % too stiff), in steps that follow the float32 precision bands; more iterations do not help. Hypothesis: the solver's small corrections fall below the float32 position step (1 µm at 10 m) and are lost. Remedy found: `env_spacing=0` stacks all envs at the origin; Newton keeps them from colliding, and all 1024 envs then give identical results and pass every check (stem alone; to re-check with the robot in M3). Question for the expert: is stacking envs at the origin the recommended practice for Newton, and are there side effects (rendering, contact buffers)?
- Gravity compensation: with `disable_gravity=False` and stiffness 400, the arm sags ~0.05 rad at joints 2 and 4 in the start pose. Disable gravity on the robot (as Isaac Lab's Franka high-PD config does), add gravity compensation, or raise the gains?
  - Leaning: do it like Isaac Lab's Franka. `FRANKA_PANDA_HIGH_PD_CFG` (`isaaclab_assets/robots/franka.py`) uses the same gains (400/80) plus `disable_gravity=True`, "useful for task-space control using differential IK". Isaac Lab's OSC how-to and gear-assembly deployment docs do the same ("Robot is mounted, no gravity"). The real Franka controller also compensates gravity itself (from memory, check in the libfranka docs).
  - Side effect: the end-effector then also has no gravity in sim. Negligible for the ~50 g `fork`, less so for the 221 g `fork_v2` (CoM 74 mm from the flange axis, about 0.16 N·m); worth knowing.

## Blocked / waiting
- Weigh the printed forks → set `inertial.measured_mass` in `fork.yaml` and `fork_v2.yaml`.

## Repo
- Add `docs/project-proposal.md` (referenced in CLAUDE.md).
