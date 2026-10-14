# Project

Semester project, ETH Zurich MSc Robotics, Systems and Control, hosted at AUTOLAB, University of Modena and Reggio Emilia: **non-prehensile manipulation of flexible plant stems with reinforcement learning.**

This is the living description of the project. The original proposal is frozen in [`proposal-original.md`](proposal-original.md); where the two differ, this file wins.

## Motivation

Plant inspection (disease, pests, phenotyping) needs views of surfaces that no fixed camera sees, such as the undersides of leaves. Getting those views means moving parts of the plant, which is done by hand today. Grasping a plant risks damage. Pushing a flexible stem moves an attached leaf into view more gently and more generally.

## Research question

> Can a reinforcement learning policy acquire a robust non-prehensile manipulation strategy that brings a selected point of a flexible stem to a target pose while respecting a hard no-damage constraint, and that generalizes across a distribution of unknown stem properties?

Sub-questions:
- How to formalize *no damage* so that it is physically meaningful and usable in a learned controller.
- How much of the stem's state the policy must observe for the task to succeed.

## Task

A robot pushes (does not grasp) a flexible stem with a 3D-printed tool so that a selected point on the stem, e.g. where a leaf is attached, reaches a target pose. The stem's mechanical properties (stiffness, length, diameter, damping) are unknown and vary per plant, so they are domain-randomized during training.

The learned policy is a low-level skill. A higher-level module will later choose which part of the plant to move and supply the target; that module is out of scope.

## Stages

1. Position control of the selected stem point.
2. Full pose control (position and orientation) of the frame attached to that point.
3. Real-robot validation on an artificial plant, with the stem state estimated from cameras.

## Scope

- **In scope:** stem model and simulation environment; the state-based policy from position to pose control and its evaluation; real-robot validation (stage 3); camera-based estimation of the stem state, developed in parallel with the first training runs (2026-10-06).
- **Stretch goal:** distilling the state-based policy into an end-to-end vision-based policy (camera images in, actions out), with the state-based policy as teacher.
- **Out of scope:** choosing the target (which part of the plant to move); a full perception or VLA system; a model-based baseline (nominal vs. oracle stem parameters): too hard to build without assuming away the non-prehensile contact (user, 2026-10-14).
- **Comparison:** the RL policy is the baseline for Simone's VLA at AUTOLAB.

## Design constraints

- **Non-prehensile contact:** intermittent, unilateral, with friction and sliding. The tool is never attached to the stem.
- **No damage is a hard limit**, enforced through terminations and penalties. Active limits: stem curvature (minimum bending radius) and robot-stem contact force. Twist and tension are measured; their limits are open.
- **Observations stay within what perception can deliver.** In simulation the policy observes the stem state directly; on the real robot it comes from cameras. Contact forces are not observed; they are used in rewards and terminations (2026-10-06).
- **Sim-to-real:** actions are in end-effector space, so the skill is not tied to one arm. The controller is Franka's Cartesian impedance law, the same code planned for the real FR3 (2026-10-07).

## Platform

- Simulation: NVIDIA Isaac Sim, Isaac Lab, PPO.
- Robot: Franka Research 3 (reference platform).
- Tool: 3D-printed fork, `fork_v2`.
- Stem model: `chain`, rigid segments with bending and twist spring joints in PhysX, sharing one solver with the robot (chosen with the supervisors, 2026-10-06). The Newton `cable` is kept for comparison.

## People

- **Simone Silenzi** (AUTOLAB, PhD student of Prof. Biagiotti): main supervisor, day to day.
- **Prof. Luigi Biagiotti** (AUTOLAB): host professor; robot and lab.
- **Mayank Mittal** (ETH Zurich, RSL): ETH supervisor; biweekly written updates; midterm 2026-10-29.
- **Prof. Marco Hutter** (ETH Zurich, RSL): home professor; little day-to-day involvement.
- **Alessio Caporali** (University of Bologna, external): stem modelling and perception.
