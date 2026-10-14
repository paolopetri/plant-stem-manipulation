> **Original proposal (draft), frozen.** Where it differs from `docs/project.md`, that file wins.

# Non-Prehensile Manipulation of Flexible Plant Stems

*A Reinforcement Learning Skill for Contact-Rich Robotic Plant Inspection*

Semester Project Proposal (Draft)

Paolo Petri — MSc Robotics, Systems and Control, ETH Zurich

Host: AUTOLAB, University of Modena and Reggio Emilia · Host supervision: Prof. Luigi Biagiotti, Simone Silenzi

## 1 Motivation

Automated plant inspection is central to modern agriculture and plant science: detecting disease, monitoring pests, and phenotyping all depend on close visual access to plant surfaces. Much of the diagnostically relevant information, however, sits on surfaces occluded from any fixed camera viewpoint — most notably the undersides of leaves. Acquiring these views requires physically reconfiguring parts of the plant, a task currently done by hand.

Plants are fragile and highly compliant, so grasping them firmly risks damage. A gentler and more general strategy is to reposition a flexible, non-woody part of the plant — a stem or branch — by pushing on it, so that an attached region of interest (for instance a leaf) is brought into view. Such a stem behaves approximately like an elastic rod, a deformable linear object: applying a load at one point transmits motion and deformation along its length. This project investigates whether a reinforcement learning (RL) policy can perform this repositioning through *non-prehensile* interaction, bringing a selected point of the stem to a target pose without mechanically damaging the plant.

The skill is intended as the low-level manipulation primitive beneath a future perception-driven system: a higher-level module would decide *which* part of the plant to move and supply the target, while the learned policy executes the physical interaction. The contribution of this project is the manipulation skill itself, together with a goal interface that such a module can later drive.

## 2 Problem Description and Research Question

**Non-prehensile interaction.** The defining constraint of this project is that the interaction is *non-prehensile*: the end-effector, a dedicated 3D-printed tool, pushes against the stem rather than grasping it. This constraint is what makes the problem both scientifically interesting and a genuine candidate for reinforcement learning. If the stem were firmly grasped, its motion would be largely kinematically determined and classical control would suffice. Non-prehensile interaction instead introduces intermittent and unilateral contact, friction and sliding, contact switching and possible loss of contact, and a strong coupling between end-effector motion and the deformation of the stem. Handling this coupling robustly, under uncertainty about the stem's mechanical properties, is the core challenge.

**Task.** We consider a flexible linear element (stem or branch) whose configuration and orientation are relevant, in part because they determine the pose of the attached leaves. A reference frame is attached to a selected point of interest on this element. The objective is to bring that frame to a prescribed target pose by applying a non-prehensile interaction with the tool at another point of the element and executing an appropriate motion. The stem's mechanical properties (stiffness, length, diameter, damping) are unknown and vary from plant to plant, and must be inferred implicitly through interaction.

**Research question.**

> Can a reinforcement learning policy acquire a robust non-prehensile manipulation strategy that brings a selected point of a flexible stem to a target pose while respecting a hard no-damage constraint, and that generalizes across a distribution of unknown stem properties?

Two sub-questions follow. First, how should *no damage* be formalized so that it is both physically meaningful and usable inside a learned controller — a maximum-curvature (minimum bending radius) constraint on the stem is the natural candidate, and it is worth checking whether comparable criteria or thresholds already exist in the literature, or must be derived experimentally or from simplified mechanical arguments. Second, how much of the stem's state must actually be observed for the task to succeed (discussed further under Observations).

## 3 Plant Modelling and Simulation

The policy is to be trained in simulation, using Isaac Sim, with Isaac Lab for the RL framework. A central premise of the project is that the quality and robustness of the learned policy depend directly on the fidelity of the stem model used for training; this modelling aspect has received comparatively little attention in the literature and may constitute a research contribution in its own right.

The stem is modelled as a deformable linear object — for example a reduced-order elastic rod, such as a chain of rigid segments coupled by elastic and torsional elements, or a comparable formulation — capturing bending and the relevant contact behaviour while remaining efficient enough for large-scale RL training. Because no measured material data exists, the model parameters are randomized during training (domain randomization); this parameter uncertainty is not an inconvenience but the core of the research question. A preliminary model will be developed with the support of a collaborator at the University of Bologna working on the manipulation of deformable linear objects (primarily cables). The specific modelling choices — the elastic formulation, the fidelity of the contact model, and validation against real stem behaviour — are to be established jointly at the start of the project.

## 4 Envisaged Approach

**Platform and tool.** The task is formulated in end-effector space so that the skill is not tied to one specific arm; the Franka Research 3 available at the host lab serves as the reference platform. The end-effector is a dedicated non-prehensile tool, designed and 3D-printed for the task, rather than the standard gripper.

**Task staging.** Following the host lab's suggestion, the task is approached in stages of increasing difficulty: first controlling only the *position* of the selected point on the stem, then extending to full *pose* (position and orientation) of the attached reference frame. The frame's orientation may additionally encode the orientation of attached leaves, depending on the plant model.

**Observations (primary objective).** The primary objective is a controller based on the state of the flexible element. In simulation this state is directly available to the policy; in a real setup it could be estimated from one or more cameras. Reducing reliance on privileged state is therefore a perception problem deliberately staged after the control problem.

**Vision-based distillation (secondary objective).** If time allows, a secondary objective is to investigate whether the state-based policy can be distilled into a *vision-based* policy that reproduces the learned behaviour from images rather than direct state — potentially within a VLA framework. This is kept clearly separate from the primary objective, as it may significantly extend the scope.

**Model-based comparison.** A meaningful comparison against model-based control is desirable, both to quantify what learning contributes and to guard against over-engineering. Constructing such a baseline is, however, itself difficult under non-prehensile contact, precisely because of the intermittent contact, friction, sliding, and contact switching noted above. We will therefore first specify the task precisely enough to determine which model-based controllers are feasible and scientifically meaningful — i.e. which can be built *without* implicitly assuming away the non-prehensile difficulties. A graded comparison, between a controller using nominal stem parameters and an oracle controller given the true per-episode parameters (available for free in simulation), is conceptually sound and remains the target, subject to this feasibility check.

## 5 Scope

**In scope:** the non-prehensile manipulation skill — plant/stem model and simulation environment, the learned state-based policy from position control up to pose control, and its evaluation, including a model-based comparison where feasible. Stretch goals, if time allows: experimental validation on the real Franka Research 3 using an artificial plant (flexible rods with artificial leaves, preferred initially for repeatability over a real plant), and vision-based distillation of the policy. **Out of scope:** a full perception or VLA system and autonomous selection of which part of the plant to move; these are the intended follow-up work that this skill is designed to serve.
