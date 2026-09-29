"""Isaac Lab config for the plant stem, modelled as a Newton cable (stem model v1).

Requirements:
- `stem_cfg() -> CableObjectCfg`: a vertical cable built from `sim_utils.CableCfg` + `CableMaterialCfg`
  (see IsaacLab `docs/source/concepts/deformables.rst`, section Cables).
- All parameters from `assets/stem/stem.yaml` (length, number of segments, diameter, density, stretch/bend
  moduli, base position). Nothing hardcoded.
- The stem base must be held fixed (clamped, or as close as the cable model allows: pinning only gives a ball
  joint, so try pinning the first two control points). Document the chosen workaround here.
- Collision enabled, so the fork can push it.
- Newton only (VBD solver); the scene must use a Newton physics cfg.

Verify: `scripts/check_stem.py` (stands, sags under gravity, springs back, deflects when pushed).
See docs/TODO.md -> M1.
"""
