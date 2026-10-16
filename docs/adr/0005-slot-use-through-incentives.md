# Slot use through incentives, not contact rules

The first M6 run (`m6-first`, 2026-10-16) brought the stem point to the bowl and stopped there: below the bowl the
stem needs a bending moment, and with fork_v2 that was out of reach within the limits. The fork applies the moment as
the slot moment: the stem passes through the slot and the fork turns about the slot's axis, so the prongs press on
opposite sides of the stem, a lever apart along it (about the prong gap where the stem sits). We keep ADR 0004's
incentive on the outcome and make the slot the natural choice without a contact rule: no termination or penalty on
prong contact or on the contact location, since pushing with a prong's outside may be the better solution for some
targets. Instead: potential-based shaping into the slot (it leaves the optimal policy unchanged and only makes the
slot easier to find), a share of resets with the stem already in the slot, a randomized fork-stem friction (a prong
push relies on friction, the slot does not), a compression limit relative to the stem's buckling load (pressing the
tip is unwanted; the axial force of slot use stays below it), and targets only where all active damage limits allow
them. Decided by the user in the grilling session of 2026-10-16; analysis in `docs/experiments/2026-10-16-m6-first.md`.

## Considered options

- **Termination or penalty on prong contact:** rejected, it hard-codes which part of the fork may push.
- **ADR 0004's fallback, a penalty on contact with the top 4 segments:** rejected, it moves the contact down the stem
  but does not make the policy use the slot.
- **Shaping towards the roll of the slot moment:** deferred; the next step if the stem enters the slot but no slot
  moment appears.
- **A softer stem until a new fork exists:** rejected, the stiffness is an unmeasured placeholder (the real plant may
  be stiffer), and a new stiffness means re-checking the chain's armature.

## Consequences

- With fork_v2 the lever is about 2 cm (the slot is 10-20 mm wide, the plate 20 mm thick): the slot moment needs
  12-15 N per contact at the placeholder stiffness to reach a quarter of the depth below the bowl, against the 5 N
  placeholder limit. Every force scales with the bending stiffness EI. The candidate fork has a wide V opening with
  inward hooks at the prong ends (the prongs push the stem towards the mouth, the hooks hold it): a 10 cm gap gives a
  10 cm lever and 3-4 N. It is sized from the stem's measured stiffness, with the supervisors. A hooked fork stays
  non-prehensile as long as the opening between the hooks is wider than the stem.
- No lever takes the stem point deeper than about 0.6 of the region below the bowl within the curvature budget; the
  deeper targets need pressing and are not sampled.
- No interim training runs on fork_v2: the next run waits for the new fork, with the damage limits unchanged (user,
  2026-10-16).
- The contact-force reading is the net force per (segment, fork) pair, so the two forces of a slot moment on one
  20 mm segment may cancel. A simulation check with the new fork measures it before the reading is changed.
