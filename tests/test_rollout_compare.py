"""Unit tests for `stem_manip.utils.rollout_compare`: the compare half of `scripts/record_rollout.py` (no simulator).

Requirements (issue #52):
- Two identical recordings compare PASS, one row per recorded term (observation, reward, termination term).
- A term that differs beyond the tolerance compares FAIL, and its row names the term and the max difference.
- A term recorded in only one file, or with another shape, compares FAIL.
- Recordings of different settings (task, envs, steps, seed, device, push rollout length) compare FAIL.
"""

from dataclasses import asdict, replace

import torch

from stem_manip.utils import check_report, rollout_compare

SETTINGS = rollout_compare.Settings(task="T", num_envs=2, steps=3, seed=1, device="cuda:0", push_steps=5)


def recording(**overrides: torch.Tensor) -> dict:
    terms = {
        "zero/obs/tool_tip_pos": torch.arange(24, dtype=torch.float32).reshape(4, 2, 3),
        "zero/reward/distance_fine": torch.full((3, 2), 0.5),
        "zero/termination/time_out": torch.tensor([[False, False], [False, True], [True, True]]),
    }
    terms.update(overrides)
    return {"meta": rollout_compare.meta(SETTINGS, commit="abc"), "terms": terms}


def test_identical_recordings_pass():
    rows = rollout_compare.compare(recording(), recording())
    assert set(rows) == {
        "settings",
        "zero/obs/tool_tip_pos",
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
    assert rows["zero/obs/tool_tip_pos"][0] and rows["zero/termination/time_out"][0]


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
    b = recording(**{"zero/obs/tool_tip_pos": torch.zeros(4, 2, 2)})
    del b["terms"]["zero/reward/distance_fine"]
    rows = rollout_compare.compare(recording(), b)
    assert not rows["zero/obs/tool_tip_pos"][0]
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
