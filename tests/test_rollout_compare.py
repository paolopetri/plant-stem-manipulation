"""Unit tests for `stem_manip.utils.rollout_compare`: the compare half of `scripts/record_rollout.py` (no simulator).

Requirements (issue #52):
- Two identical recordings compare PASS, one row per recorded term (observation, reward, termination term).
- A term that differs beyond the tolerance compares FAIL, and its row names the term and the max difference.
- A term recorded in only one file, or with another shape, compares FAIL.
- Recordings of different settings (task, envs, steps, seed, device, push rollout length) compare FAIL.
- `never_nonzero` names the terms (of those required) that stay 0 / never fire or are missing, so `record` can fail
  when the push stops exercising the damage limits.
- With renames `(old, new)` (issue #135), a term renamed between the recordings compares PASS under its new name
  and FAIL without the rename. A name matches a term key whole or as its tail after a `/` (`slot_base_pos` or
  `obs/slot_base_pos`), in every rollout. The renames apply at once, so a swap works whatever their order. An old
  name not in the baseline, a new name not in the other recording, an old name given twice, a term matched by two
  old names, or a renamed term that collides with another baseline term is an error, so no term drops out of the
  comparison unnoticed.
- The contact-sensor force terms (`<rollout>/reward/contact_force`, `<rollout>/state/robot_contact_force`, matched
  exactly below the rollout) pass within a noise floor of 1e-6 and are marked "float32 noise" when their difference
  is non-zero; every other term stays bit-exact (issue #150). An explicit `tol` applies to all terms.
"""

from dataclasses import asdict, replace

import pytest
import torch

from stem_manip.utils import check_report, rollout_compare

SETTINGS = rollout_compare.Settings(task="T", num_envs=2, steps=3, seed=1, device="cuda:0", push_steps=5)


def recording(**overrides: torch.Tensor) -> dict:
    terms = {
        "zero/obs/slot_base_pos": torch.arange(24, dtype=torch.float32).reshape(4, 2, 3),
        "zero/reward/distance_fine": torch.full((3, 2), 0.5),
        "zero/termination/time_out": torch.tensor([[False, False], [False, True], [True, True]]),
    }
    terms.update(overrides)
    return {"meta": rollout_compare.meta(SETTINGS, commit="abc"), "terms": terms}


def test_identical_recordings_pass():
    rows = rollout_compare.compare(recording(), recording())
    assert set(rows) == {
        "settings",
        "zero/obs/slot_base_pos",
        "zero/reward/distance_fine",
        "zero/termination/time_out",
    }
    assert check_report.report(rows) is True


def test_differing_term_fails_and_is_named():
    changed = torch.full((3, 2), 0.5)
    changed[1, 0] = 0.75
    rows = rollout_compare.compare(recording(), recording(**{"zero/reward/distance_fine": changed}))
    assert check_report.report(rows) is False
    ok, info = rows["zero/reward/distance_fine"]
    assert not ok and "0.25" in info
    assert rows["zero/obs/slot_base_pos"][0] and rows["zero/termination/time_out"][0]


def test_termination_flip_fails():
    flipped = torch.tensor([[False, False], [False, False], [True, True]])
    rows = rollout_compare.compare(recording(), recording(**{"zero/termination/time_out": flipped}))
    assert not rows["zero/termination/time_out"][0]


def test_difference_within_tolerance_passes():
    nudged = torch.full((3, 2), 0.5 + 1e-6)
    rows = rollout_compare.compare(recording(), recording(**{"zero/reward/distance_fine": nudged}), tol=1e-5)
    assert rows["zero/reward/distance_fine"][0]
    rows = rollout_compare.compare(recording(), recording(**{"zero/reward/distance_fine": nudged}), tol=0.0)
    assert not rows["zero/reward/distance_fine"][0]


def test_missing_term_or_other_shape_fails():
    b = recording(**{"zero/obs/slot_base_pos": torch.zeros(4, 2, 2)})
    del b["terms"]["zero/reward/distance_fine"]
    rows = rollout_compare.compare(recording(), b)
    assert not rows["zero/obs/slot_base_pos"][0]
    assert not rows["zero/reward/distance_fine"][0]


def test_other_settings_fail_commit_does_not_count():
    b = recording()
    b["meta"]["commit"] = "def"
    assert rollout_compare.compare(recording(), b)["settings"][0]
    for key in asdict(SETTINGS):
        b = recording()
        b["meta"] = rollout_compare.meta(replace(SETTINGS, **{key: "other"}), commit="abc")
        ok, info = rollout_compare.compare(recording(), b)["settings"]
        assert not ok and key in info


def test_other_push_length_fails():
    b = recording()
    b["meta"] = rollout_compare.meta(replace(SETTINGS, push_steps=6), commit="abc")
    ok, info = rollout_compare.compare(recording(), b)["settings"]
    assert not ok and "push_steps 5 vs 6" in info


def renamed(old: str, new: str) -> dict:
    """`recording()` with every term key ending in `old` renamed to end in `new`."""
    b = recording()
    b["terms"] = {(k[: -len(old)] + new if k.endswith(old) else k): v for k, v in b["terms"].items()}
    return b


def test_renamed_term_passes_with_rename_fails_without():
    b = renamed("slot_base_pos", "slot_base_position")
    assert check_report.report(rollout_compare.compare(recording(), b)) is False
    rows = rollout_compare.compare(recording(), b, renames=[("slot_base_pos", "slot_base_position")])
    assert check_report.report(rows) is True
    assert "zero/obs/slot_base_position" in rows


def test_rename_matches_key_tail_after_slash():
    b = renamed("reward/distance_fine", "reward/distance_near")
    rows = rollout_compare.compare(recording(), b, renames=[("reward/distance_fine", "reward/distance_near")])
    assert check_report.report(rows) is True
    # "fine" is not a whole name after a "/": no match
    with pytest.raises(ValueError, match="'fine' matches no term of the first"):
        rollout_compare.compare(recording(), b, renames=[("fine", "near")])


def test_rename_name_not_in_recordings_is_an_error():
    b = renamed("slot_base_pos", "slot_base_position")
    with pytest.raises(ValueError, match="'slot_base_po' matches no term of the first"):
        rollout_compare.compare(recording(), b, renames=[("slot_base_po", "slot_base_position")])
    with pytest.raises(ValueError, match="'slot_base_posit' matches no term of the second"):
        rollout_compare.compare(recording(), b, renames=[("slot_base_pos", "slot_base_posit")])


def two_terms(x: torch.Tensor, y: torch.Tensor) -> dict:
    """A recording with the terms `zero/obs/x` and `zero/obs/y`."""
    return {"meta": rollout_compare.meta(SETTINGS, commit="abc"), "terms": {"zero/obs/x": x, "zero/obs/y": y}}


def test_swap_passes_in_either_order():
    a, b = two_terms(torch.zeros(3), torch.ones(3)), two_terms(torch.ones(3), torch.zeros(3))
    for renames in ([("x", "y"), ("y", "x")], [("y", "x"), ("x", "y")]):
        assert check_report.report(rollout_compare.compare(a, b, renames=renames)) is True


def test_renamed_term_colliding_with_a_baseline_term_is_an_error():
    a = two_terms(torch.zeros(3), torch.ones(3))
    b = {"meta": a["meta"], "terms": {"zero/obs/y": torch.zeros(3)}}
    with pytest.raises(ValueError, match="'zero/obs/y'"):
        rollout_compare.compare(a, b, renames=[("x", "y")])


def test_old_name_twice_or_term_matched_twice_is_an_error():
    a = two_terms(torch.zeros(3), torch.ones(3))
    b = {"meta": a["meta"], "terms": {"zero/obs/u": torch.zeros(3), "zero/obs/v": torch.zeros(3)}}
    with pytest.raises(ValueError, match="'x' is renamed twice"):
        rollout_compare.compare(a, b, renames=[("x", "u"), ("x", "v")])
    with pytest.raises(ValueError, match="'zero/obs/x' is matched by both 'x' and 'obs/x'"):
        rollout_compare.compare(a, b, renames=[("x", "u"), ("obs/x", "v")])


def test_never_nonzero_names_silent_and_missing_terms():
    terms = recording()["terms"]
    terms["push/reward/curvature"] = torch.zeros(3, 2)
    required = ["zero/reward/distance_fine", "zero/termination/time_out", "push/reward/curvature", "push/x"]
    assert rollout_compare.never_nonzero(terms, required) == ["push/reward/curvature", "push/x"]
    assert rollout_compare.never_nonzero(terms, required[:2]) == []


def contact_recording(diff: float) -> dict:
    """`recording()` plus the push contact-force terms, offset by `diff` [N]."""
    force = torch.tensor([[0.0, 1.5], [2.0, 0.0], [3.0, 4.0]], dtype=torch.float64) + diff
    return recording(**{"push/state/robot_contact_force": force, "push/reward/contact_force": -force})


def test_contact_force_noise_passes_as_float32_noise():
    rows = rollout_compare.compare(contact_recording(0.0), contact_recording(2.4e-7))
    assert check_report.report(rows) is True
    for name in ("push/state/robot_contact_force", "push/reward/contact_force"):
        ok, info = rows[name]
        assert ok and "float32 noise" in info


def test_equal_contact_force_is_not_marked_as_noise():
    rows = rollout_compare.compare(contact_recording(0.0), contact_recording(0.0))
    assert "float32 noise" not in rows["push/state/robot_contact_force"][1]


def test_contact_force_beyond_noise_floor_fails():
    rows = rollout_compare.compare(contact_recording(0.0), contact_recording(2e-6))
    assert not rows["push/state/robot_contact_force"][0]
    assert not rows["push/reward/contact_force"][0]


def test_other_terms_stay_bit_exact():
    nudged = torch.full((3, 2), 0.5, dtype=torch.float64) + 1e-9
    rows = rollout_compare.compare(recording(), recording(**{"zero/reward/distance_fine": nudged}))
    assert not rows["zero/reward/distance_fine"][0]


def test_explicit_tol_applies_to_contact_force_too():
    rows = rollout_compare.compare(contact_recording(0.0), contact_recording(2.4e-7), tol=0.0)
    assert not rows["push/state/robot_contact_force"][0]
    rows = rollout_compare.compare(contact_recording(0.0), contact_recording(2e-6), tol=1e-5)
    ok, info = rows["push/state/robot_contact_force"]
    assert ok and "float32 noise" not in info


def test_noise_floor_matches_contact_terms_only_below_the_rollout():
    force = torch.zeros(3, 2)
    a = recording(**{"push/obs/state/robot_contact_force": force})
    b = recording(**{"push/obs/state/robot_contact_force": force + 2.4e-7})
    assert not rollout_compare.compare(a, b)["push/obs/state/robot_contact_force"][0]
