"""Unit tests for `stem_manip.utils.rollout`: helpers of the scripted rollouts in the checks (no simulator).

Requirements (issue #52, review of PR #109):
- `obs_term_slices`: the slice of each term in the concatenated observation, in order, multi-dim terms flattened.
- `agent_action`: zero agent all zeros; random agent uniform in [-1, 1], the same actions for the same seed.
- `toward`: the translation action toward a goal, at most `speed` x the step cap, braking for the step-change limit
  (step size sqrt(2 ds d) at distance d); rotation part 0.
"""

import math

import pytest
import torch

from stem_manip.utils import rollout


def test_obs_term_slices_in_order_and_flattened():
    slices = rollout.obs_term_slices(["a", "b", "c"], [(3,), (2, 5), (1,)])
    assert slices == {"a": slice(0, 3), "b": slice(3, 13), "c": slice(13, 14)}


def test_zero_agent_is_zero():
    action = rollout.agent_action("zero", 4, 6, torch.Generator().manual_seed(0))
    assert torch.equal(action, torch.zeros(4, 6))


def test_random_agent_in_range_and_seeded():
    a = rollout.agent_action("random", 64, 6, torch.Generator().manual_seed(7))
    b = rollout.agent_action("random", 64, 6, torch.Generator().manual_seed(7))
    assert a.shape == (64, 6) and torch.equal(a, b)
    assert a.min() >= -1.0 and a.max() <= 1.0 and a.min() < -0.5 and a.max() > 0.5


def test_toward_far_goal_full_speed():
    action = rollout.toward(torch.zeros(1, 3), torch.tensor([[0.0, 1.0, 0.0]]), max_step=0.0032, max_step_change=8e-5)
    assert torch.allclose(action, torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]))
    half = rollout.toward(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0]]), 0.0032, 8e-5, speed=0.5)
    assert half[0, 0].item() == pytest.approx(0.5)


def test_toward_near_goal_brakes():
    # distance 0.1 mm: step sqrt(2 x 0.08 mm x 0.1 mm) = 0.1265 mm = 0.03953 x the 3.2 mm cap
    action = rollout.toward(torch.zeros(1, 3), torch.tensor([[0.0, 0.0, -1e-4]]), 0.0032, 8e-5)
    assert action[0, 2].item() == pytest.approx(-math.sqrt(2 * 8e-5 * 1e-4) / 0.0032, rel=1e-5)
    assert action[0, 2].item() == pytest.approx(-0.03953, rel=1e-3)
    assert torch.equal(action[0, :2], torch.zeros(2)) and torch.equal(action[0, 3:], torch.zeros(3))


def test_toward_at_goal_is_zero():
    action = rollout.toward(torch.ones(2, 3), torch.ones(2, 3), 0.0032, 8e-5)
    assert torch.equal(action, torch.zeros(2, 6))
