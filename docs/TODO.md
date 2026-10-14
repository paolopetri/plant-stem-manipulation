# TODO

High-level items and discussion: current milestone, roadmap, research directions, open questions per person. Concrete, actionable work items are GitHub issues (decision 2026-10-13). When an item here is done or turned into issues, delete it and log it in today's `docs/notes/` file.

## Current milestone: M5 First training run
Stem model `chain` in PhysX (decision 2026-10-06). Exit criterion: tag `v0.1-position-control` once a policy reaches the targets.
- Current work: #77, #78, #79, #80, #81, #82, #84, #85, #86, #88, #94, #95.
- Later (closed as not planned to keep the open list to current work, 2026-10-14; the full text stays in each issue, reopen when it becomes current):
  - #75 Two targets in sequence within an episode
  - #76 Refine the stem spawn area near the robot
  - #83 Port the impedance controller to a libfranka 1 kHz loop
  - #87 Contact forces as critic-only observations
  - #89 Choose the randomization ranges for the stem parameters
  - #90 Per-env stem parameters in the chain and the MDP terms
  - #91 Tilted stem at reset
  - #92 Real-robot identification of the FR3
  - #93 Spawn the stem all around the robot
  - #100 B3: inertia feedforward (operational-space control)
  - #101 Re-checks over the stem randomization range
  - #103 Stage 2: full pose control of the attached frame
  - #104 Model-based baseline (nominal vs. oracle parameters)

## Roadmap
1. Domain randomization of the stem parameters (stiffness, length, diameter, damping): ranges #89, per-env parameters #90, re-checks over the range #101.
2. Stage 2: full pose control (#103).
3. Model-based baseline (nominal vs. oracle parameters, #104).
4. Real-robot validation on an artificial plant (identification #92, controller port #83).
5. Distillation into a vision-based policy.

## Research directions
- **Vision model with Alessio Caporali:** set up a vision model that estimates the stem state from camera images. Why: Alessio's experience is that RL policies have trouble transferring to the real world, and the stem state the policy observes must come from cameras there. What it must deliver: #88.
- **Specialize the policy for the test plant**, most likely a thin bamboo (Alessio's suggestion, 2026-10-06): after broad training, condition the policy on stem parameters close to the real plant. Options: (A) fine-tune with the randomization narrowed around the measured values; (B) the stem parameters as an extra policy input, with the measured values entered on the real robot; (C) online adaptation from the motion history. Needs the bamboo's length, diameter, bending stiffness (cantilever test) and damping (decay after a flick); re-check the chain's armature for these values.
- **Branched plants** (user, 2026-10-09): add branches to the plant model (e.g. a PhysX articulation tree instead of the chain). The target command assumes one uniform, upright, inextensible rod with loads only at the stem point (`stem_manip.utils.stem_target`: bowl, drop and curvature factors; rest position straight above the base). With branches the rest shape bends under their weight, stiffness varies at the junctions, a stem point on a branch has a kinked path to the base, and branches can catch on the fork. Candidate: targets from a simulated library (random forces / moments, keep end positions with every joint within the curvature budget). Also check the stem interface (`segment_poses` as a chain), the observed points along the stem, and the damage measures per joint (should carry over).

## Open question: damage criteria beyond bending
For all supervisors and Alessio. The measures exist (twist rate, axial strain, contact force), but only the bending curvature has a limit. Issues waiting for values: #81 (twist), #82 (tension).
- **Torsion:** the fork can turn (6-D action) and twist a stem caught in the slot; `damage.max_twist_rate` is null.
- **Tension:** tearing is limited as a tensile stress (`damage.max_tensile_stress`, null), e.g. the fork dragging along the stem with friction.
- **Crushing / abrasion:** needs a contact-force limit. `damage.max_contact_force` 5 N is a placeholder: a full-speed slot impact at 0.25 m height gives only 1.07 N (mean over a policy step), so with the 4 N clamp it fires only in unusual cases (e.g. arm contact).
- **Rigid segments and local loads** (user, 2026-10-05): with rigid segments the fork acts on a rigid body part, so a twisting or crushing load is spread over the segment and its neighbouring joints, while a real stem deforms and can break locally. The policy could twist or press a segment hard without being punished, so the limits must cover the twist and the contact force, not only the bending curvature.

Realistic values: literature, or measurements on the artificial plant?

## Questions for Alessio Caporali (stem model and perception)
Collected for one email.
- **Stem-state observation:** the policy observes a subsample of points along the stem (decision 2026-10-06). How many points along the stem can the vision model deliver, how accurately (mm), at what rate, and with what latency? Can it identify the tip (the stem point the policy controls)? The policy observes the stem base and 5 points at 0.08 ... 0.40 m, at the current and the previous policy step (32 ms earlier): can it see the base (where the stem leaves the pot), and deliver two consecutive frames at about 31 Hz? (#88)
- **Specialization to the test plant:** which of options A/B/C (Research directions) does he mean? Likely A, possibly with B.
- **Stem damping:** with `damping_time` 3.2 ms (damping ratio about 0.06) the stem swings several times after a kick, more like a bamboo stick than a living stem, which is likely much more damped. Can we measure the decay on the artificial plant? (#89)
- **Chain joint armature** (2026-10-05): PhysX solves the joint springs iteratively, and without armature the 1 g segments on E I / l springs come out 16 x too soft (8 iterations; 128 iterations still 4 x). Armature 1e-4 kg m² per joint coordinate fixes the static shape and the first mode (-1 %) but makes modes 2 and 3 26 % and 64 % too slow. Is that acceptable for slow pushing, or is a solver without this problem preferable (e.g. MuJoCo-Warp, which solves joint-space dynamics with a direct solver; untested)?

## Questions for Luigi Biagiotti (robot and lab)
- **Robot controller** (`docs/overleaf_folder/open_questions/impedance_action_study.tex`): is the proposed controller (Franka's Cartesian impedance law with apparent-mass damping, K_p 1000 N/m, K_o 100 N m/rad, at most 10 cm/s and 45 deg/s, contact force bounded to 4 N) acceptable next to a plant? (Port: #83.)
- **Action limits** (`docs/overleaf_folder/open_questions/action_limits_problem.tex`, Section "Questions", 6 questions): apparent mass and motor inertia on the FR3 (does the joint torque controller reduce it; which mass matrix libfranka uses), stiffness K 1000 / K_o 100 in a libfranka torque loop and velocity filtering, accuracy of libfranka's model for inertia feedforward (option B3, #100), whether 0.08 m/s^2 is reasonable near a plant, which reflexes and thresholds the lab configures (#79).
- **Real-robot setup:** is the Franka Control Interface (FCI, torque control) enabled on our FR3; which system and libfranka versions; is there a PC with a real-time kernel connected to it; does the lab already use a control stack (franka_ros2, frankapy, own C++) that the controller should be added to?

## Parked: Newton cable path
Stem model `chain` chosen 2026-10-06; `cable` stays selectable for a later comparison. Issues (label `parked`, closed as not planned 2026-10-14; the text stays in each issue): #96 base-placement function for the stem models, #97 former M2: FR3 on Newton, #98 former M3: coupled scene, #99 questions if the cable comes back. Solver measurements: `docs/overleaf_folder/open_questions/stem_model_tradeoff.tex`.
