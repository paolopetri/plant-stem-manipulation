"""The defaults of the push action cfg equal the decided values (guards against silent changes; no simulator).

Source of the values: docs/overleaf_folder/open_questions/action_limits_problem.tex (option B1, user 2026-10-07/08;
speed cap and acceleration limit revised 2026-10-08 after the full-speed sweep and the real reversals).
"""

import pytest

from stem_manip.tasks.push_position.mdp.actions_cfg import ToolTipImpedanceActionCfg

DECIDED = {
    "max_step": 0.0032,  # [m] speed cap 10 cm/s at 31.25 Hz (2026-10-08; 20 cm/s needs 21 cm to reach)
    "max_rot_step": 0.02513,  # [rad] 45 deg/s (2026-10-07)
    "max_step_change": 0.00008,  # [m] 0.08 m/s^2: steady lag Lambda a / K <= 1.72 mm at Lambda 21.5 kg (2026-10-08)
    "max_rot_step_change": 0.000349,  # [rad] 0.02 deg/step = 20 deg/s^2 (table tab:al-b1)
    "max_target_offset": 0.004,  # [m] clamp, 4 N at K_p 1000 (2026-10-08)
    "max_target_rot_offset": 0.05236,  # [rad] clamp 3 deg, 5.2 N m at K_o 100 < FR3 wrist limit 12 N m (2026-10-08)
    "stiffness_pos": 1000.0,  # [N/m] option B1 (2026-10-07)
    "stiffness_rot": 100.0,  # [N m/rad] tilt <= 1 deg at the B1 acceleration (table tab:al-diag)
}


@pytest.mark.parametrize("field", DECIDED)
def test_cfg_default_is_decided_value(field: str):
    """Each limit and gain of the cfg defaults to the decided value."""
    assert getattr(ToolTipImpedanceActionCfg(), field) == pytest.approx(DECIDED[field], rel=1e-9)
