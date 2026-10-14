# Contact forces are not observed

The policy does not observe contact forces between the robot and the stem; rewards and terminations may use them.
On the real robot the observations come from cameras, and there is no flange force-torque sensor. The FR3's own
estimate of the external wrench may not resolve the 0.6-1.6 N of a push. A policy that relied on the forces could
not be deployed. Decided with the supervisors on 2026-10-06 (`docs/notes/2026-10-06.md`).

## Consequences

- The contact-force limit and the force penalty are enforced only in simulation; on the real robot the policy has
  to stay within them through what it learned.
- Contact forces may still reach the critic as privileged observations (asymmetric actor-critic), never the actor.
