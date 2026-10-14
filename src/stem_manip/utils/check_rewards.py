"""Reward classification of `scripts/check_push_agents.py`, importable without the simulator.

An independent copy of the push task cfg's reward terms (`CODING_STANDARDS.md`, single source of truth);
`tests/test_check_push_agents_rewards.py` guards that the names match the cfg.
"""

POSITIVE_REWARDS = ("distance_coarse", "distance_fine", "height", "approach")  # tanh terms in [0, weight]
PENALTIES = ("curvature", "contact_force", "action_rate", "terminated")
ZERO_PENALTIES = ("curvature", "contact_force", "action_rate")  # zero agent: exactly 0
