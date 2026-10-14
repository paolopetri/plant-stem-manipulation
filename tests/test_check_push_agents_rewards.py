"""The reward classification of `scripts/check_push_agents.py` covers exactly the push task cfg's reward terms (no
simulator). The lists stay an independent copy of the cfg (`CODING_STANDARDS.md`, single source of truth); this only
guards that their names match. Terms of weight 0 count: Isaac Lab leaves them out of the reward manager's active terms,
so the check must not infer its terms from there.
"""

from isaaclab.managers import RewardTermCfg

from stem_manip.tasks.push_position.env_cfg import StemPushPositionEnvCfg
from stem_manip.utils.check_rewards import PENALTIES, POSITIVE_REWARDS, ZERO_PENALTIES


def test_every_reward_term_is_in_exactly_one_list():
    cfg = StemPushPositionEnvCfg()
    declared = [name for name, value in vars(cfg.rewards).items() if isinstance(value, RewardTermCfg)]
    classified = [*POSITIVE_REWARDS, *PENALTIES]
    assert sorted(classified) == sorted(declared)  # none missing, none stale, none in both lists


def test_zero_penalties_are_penalties():
    assert set(ZERO_PENALTIES) <= set(PENALTIES)
