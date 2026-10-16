# Slot use through target choice, not a hard-coded contact location

The first training run (M5, 2026-10-10) pushed the stem at its tip with a prong's side and never took the stem into
the slot; it stalled a few mm above deep targets, because the targets lay at most 5-8 mm below the bowl. We make the
policy use the slot by giving it targets it can only reach with a moment or a C/S bend, not by rewarding or
penalizing where the fork touches the stem: targets are sampled from the reachable region of a large-deflection rod
model, down to the deepest shape within the curvature budget, and the start pose is varied so that first contacts
happen along the whole stem. The incentive stays on the outcome (the stem point at the target, without damage), so
the policy finds its own way of using the fork; the same task carries over to stage 2, where the stem point is not
the tip and contact near it may be right. Decided by the user in the grilling session of 2026-10-15.

## Considered options

- **Penalty on contact with the top k segments:** rejected as the first step, it hard-codes where to push. Kept as
  the fallback if the contact-point metric and the play videos still show tip-only pushing. Superseded by ADR 0005
  (2026-10-16): not run, it moves the contact but does not make the policy use the slot.
- **Linear (small-deflection) target formulas with wider ranges:** wrong at the depths needed.

## Consequences

- In large deflection a force alone can also reach deep targets, by pressing the tip down (axial compression,
  buckling). The axial compression is logged, and a compression limit (a damage mode next to tension, #82) is decided
  from the first run's data.
