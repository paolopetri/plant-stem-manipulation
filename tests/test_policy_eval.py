"""`policy_eval` summarizes the episodes of a policy evaluation (no simulator).
Hand-made episodes with known shares, errors and angles.
"""

import pytest
import torch

from stem_manip.utils import policy_eval


def episodes(**overrides) -> policy_eval.Episodes:
    """Four episodes: two time-outs, one curvature limit, one ended by contact force and joint margin at once."""
    fields = dict(
        ended_by={
            "time_out": torch.tensor([True, True, False, False]),
            "curvature_limit": torch.tensor([False, False, True, False]),
            "contact_force_limit": torch.tensor([False, False, False, True]),
            "joint_margin": torch.tensor([False, False, False, True]),
        },
        length_s=torch.tensor([15.0, 15.0, 3.0, 5.0]),
        position_error=torch.tensor([0.005, 0.015, 0.04, 0.08]),
        height_error=torch.tensor([0.002, -0.004, 0.03, 0.06]),
        max_rotation=torch.tensor([0.1, 0.3, 0.5, 1.5]),
        final_rotation=torch.tensor([0.1, 0.2, 0.5, 1.0]),
    )
    fields.update(overrides)
    return policy_eval.Episodes(**fields)


def test_shares_of_how_episodes_end():
    summary = policy_eval.summarize(episodes(), time_out="time_out", success_distance=0.01)
    assert summary["episodes"] == 4
    assert summary["length_s/mean"] == pytest.approx(9.5)
    assert summary["ended_by/time_out"] == pytest.approx(0.5)
    assert summary["ended_by/curvature_limit"] == pytest.approx(0.25)
    assert summary["ended_by/contact_force_limit"] == pytest.approx(0.25)
    assert summary["ended_by/joint_margin"] == pytest.approx(0.25)
    # two early terms in one episode count once
    assert summary["ended_by/early (any)"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "group, expected",
    [
        # sorted position errors 0.005 0.015 0.04 0.08; height errors -0.004 0.002 0.03 0.06
        (
            "all",
            {
                "position_error/mean": 0.035,
                "position_error/median": 0.0275,
                "position_error/p90": 0.068,
                "reached": 0.25,
                "height_error/mean": 0.022,
                "height_error/median": 0.016,
                "height_error/p10": -0.0022,
                "height_error/p90": 0.051,
            },
        ),
        # the two time-out episodes: position errors 0.005 0.015, height errors -0.004 0.002
        (
            "time_out",
            {
                "position_error/mean": 0.01,
                "position_error/median": 0.01,
                "position_error/p90": 0.014,
                "reached": 0.5,
                "height_error/mean": -0.001,
                "height_error/median": -0.001,
                "height_error/p10": -0.0034,
                "height_error/p90": 0.0014,
            },
        ),
    ],
)
def test_final_errors_over_all_and_time_out_episodes(group, expected):
    summary = policy_eval.summarize(episodes(), time_out="time_out", success_distance=0.01)
    for key, value in expected.items():
        assert summary[f"{group}/{key}"] == pytest.approx(value), key


def test_fork_rotation_over_all_episodes():
    # sorted largest angles 0.1 0.3 0.5 1.5; final angles 0.1 0.2 0.5 1.0
    summary = policy_eval.summarize(episodes(), time_out="time_out", success_distance=0.01)
    expected = {
        "max_rotation/mean": 0.6,
        "max_rotation/median": 0.4,
        "max_rotation/p90": 1.2,
        "final_rotation/mean": 0.45,
        "final_rotation/median": 0.35,
        "final_rotation/p90": 0.85,
    }
    for key, value in expected.items():
        assert summary[f"all/{key}"] == pytest.approx(value), key


def test_no_time_out_episodes_leave_out_the_time_out_group():
    ended_early = episodes().ended_by | {"time_out": torch.tensor([False] * 4)}
    summary = policy_eval.summarize(episodes(ended_by=ended_early), time_out="time_out", success_distance=0.01)
    assert summary["ended_by/time_out"] == 0.0
    assert "all/position_error/mean" in summary
    assert not [key for key in summary if key.startswith("time_out/")]
