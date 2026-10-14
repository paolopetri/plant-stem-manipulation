# Franka's Cartesian impedance law as the action

An action is a small step of the slot-base pose in the robot base frame. The step moves the setpoint, and Franka's
Cartesian impedance law tracks it with joint torques. This is the controller planned for the real FR3, so the same
code runs in simulation and on the robot, and the skill is defined in end-effector space rather than for one arm.
Isaac Lab's differential IK or joint-position control would be simpler in simulation, but would not behave like the
real robot's controller. Speed, acceleration and setpoint limits bound the contact force. Decided on 2026-10-07
(`docs/notes/2026-10-07.md`); limits and their reasons: `docs/overleaf_folder/open_questions/action_limits_problem.tex`.

## Considered Options

- Feedback only, lag bounded by the acceleration limit (chosen).
- Inertia feedforward (operational-space control, Khatib): smaller lag, but the motion depends on the model of the
  apparent mass. Revisited only after a successful training and the identification on the real robot.

## Consequences

- The arm is torque controlled in simulation, as on the real FR3.
- Rotation is coupled to translation; decoupling them (full operational-space control) is deferred until after the
  first transfer to the real robot.
