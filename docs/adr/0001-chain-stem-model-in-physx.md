# Stem model `chain` in PhysX instead of the Newton `cable`

Training uses the `chain` stem model: rigid segments joined by bending and twist springs, a PhysX articulation that
shares one solver with the robot. The Newton `cable` models stretch and shear as well, but it runs only on Newton,
so the whole scene, robot included, would have to move to Newton and the fork would reach the cable only as a proxy
collider in a coupled solve. That path still had open questions on base clamping, solver convergence, per-env
randomization and contact sensors (#96-#99). The `chain` reads contact forces with a standard contact sensor and
matched the stem's stiffness within about 5 % in the contact check. Decided with the supervisors and Alessio
Caporali on 2026-10-06 (`docs/notes/2026-10-06.md`).

## Considered Options

- Newton `cable`, coupled to the robot: parked (#96-#99), kept selectable for a later comparison.
- A quasi-static, purely elastic model (Alessio Caporali's suggestion): open as a comparison or baseline.
- Full comparison: `docs/overleaf_folder/open_questions/stem_model_comparison.tex`.

## Consequences

- The chain cannot stretch, so tension comes from the joint reaction forces, not from a strain.
- The MDP reads the stem through one accessor so that the stem model stays swappable (`docs/architecture.md`,
  "Stem interface").
