# Plant-stem manipulation

A robot pushes a flexible plant stem with a 3D-printed fork, without grasping it and without damaging it, so that a
selected point on the stem reaches a target.

## Language

### Stem

**Stem**:
The real or artificial flexible plant stem that the robot pushes.
_Avoid_: plant (for the stem alone)

**Stem parameters**:
The stem's mechanical properties (stiffness, length, diameter, damping), unknown per plant and randomized in training.
_Avoid_: stem properties, material constants

**Stem model**:
One selectable simulation representation of the stem, such as the rigid-segment chain or the cable.
_Avoid_: cable (as a generic word for the stem)

**Segment**:
One rigid piece of the discretized stem, counted from the base; the k segments next to the tip are the top k
segments, e.g. "top 4".
_Avoid_: first k segments (for the ones at the tip)

**Stem joint**:
The bending and twisting connection between two neighbouring segments.
_Avoid_: joint (unqualified)

**Tip**:
The free top end of the stem.
_Avoid_: top, free end

**Stem point**:
The selected point on the stem that the policy must bring to the target, e.g. where a leaf is attached; in stage 1 it
is the tip.
_Avoid_: point of interest, target point, tip

**Stem-point frame**:
The frame fixed to the stem at the stem point, e.g. along a leaf's orientation; its pose is controlled in stage 2.
_Avoid_: attached frame, leaf frame

**Rest position**:
The position of the stem point in the unloaded, upright stem.
_Avoid_: neutral position

**Bowl**:
The surface the stem point traces when the stem is pushed sideways at the point by a horizontal force alone, without a moment from the slot.
_Avoid_: reach surface

**C/S bend**:
A stem shape that needs a moment from the slot and brings the stem point below the bowl.

### Task

**Target**:
The position (stage 1) or pose (stage 2) the stem point must reach, sampled once per episode.
_Avoid_: goal, command, target (for the controller's setpoint)

**Spawn area**:
The region of the plate where the stem base is placed at each reset.

**Start pose**:
The robot's configuration at reset.
_Avoid_: home pose, initial pose

### Robot and control

**Fork**:
The 3D-printed tool on the robot flange that pushes the stem.
_Avoid_: tool, end-effector, gripper

**Prong**:
One of the fork's two arms.
_Avoid_: tine, finger

**Slot**:
The gap between the prongs, in which the stem can sit.
_Avoid_: notch, gap

**Robot joint**:
A joint of the arm.
_Avoid_: joint (unqualified)

**Setpoint**:
The pose the impedance controller pulls the fork toward; the actions move it.
_Avoid_: target, target pose, impedance target

**Slot base**:
The point at the closed end of the slot, where the prongs meet, with the fork's orientation; actions, observations and
the setpoint refer to its pose.
_Avoid_: tool tip, TCP, fork tip, reference point

**Contact point**:
Where the fork touches the stem, given as the arc length from the stem base.
_Avoid_: attach point, attachment

**Fork contact**:
Contact of the fork with the stem above a small force; the contact point is taken over the policy steps with fork
contact.
_Avoid_: touching, tool contact

**Slot contact**:
Fork contact in which the stem passes through the slot, between the prongs, and touches a slot wall or the slot base.

**Prong contact**:
Fork contact other than slot contact: on a prong's outer side, tip or flat face, or with the stem lying along the slot
and its end pressed against the slot base.
_Avoid_: side push (for the contact)

**Slot moment**:
The bending moment the fork applies to the stem passing through the slot when it turns about the slot's axis, so that
one prong presses on the stem from one side and the other prong from the opposite side, some distance apart along it.
_Avoid_: twist (that is the stem's torsion)

**Arm contact**:
Contact of any robot body other than the fork with the stem; it ends the episode. A contact rule, not a damage limit.
_Avoid_: robot contact

**Policy step**:
One action interval of the policy.
_Avoid_: control step, env step

**Physics step**:
One step of the physics solver; several make one policy step.
_Avoid_: sim step, substep

**Commanded step**:
The setpoint motion the policy asks for in one policy step.
_Avoid_: raw action

**Applied step**:
The setpoint motion actually carried out in one policy step, after the limits.

**Setpoint offset**:
The setpoint pose relative to the slot base's current pose.
_Avoid_: target offset

### Damage

**Damage mode**:
One way the stem can be damaged: bending, torsion, tearing or crushing.

**Damage limit**:
A hard threshold on a damage measure that ends the episode.

**Curvature limit**:
The damage limit on the stem's bending curvature.
_Avoid_: minimum bending radius (as the name of the limit)

**Contact-force limit**:
The damage limit on the force the robot applies to the stem.

**Curvature budget**:
The share of the curvature limit the task may use: targets stay inside it and the curvature penalty starts at it.
_Avoid_: soft limit, soft fraction
