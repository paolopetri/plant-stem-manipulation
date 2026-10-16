"""`scripts/affected_checks.py` maps changed paths to the sim checks that must be re-run on a merged head (#179)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "affected_checks.py"
spec = importlib.util.spec_from_file_location("affected_checks", SCRIPT)
affected_checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(affected_checks)

# The paths #163 changed (0de9ddb..ebae378): its start-pose change broke check_push_stem in the #160 run.
PATHS_163 = [
    "docs/architecture.md",
    "scripts/check_fr3.py",
    "scripts/check_push_agents.py",
    "scripts/check_push_env.py",
    "scripts/check_push_terms.py",
    "scripts/start_poses.py",
    "scripts/sweep_action_poses.py",
    "src/stem_manip/tasks/push_position/env_cfg.py",
    "src/stem_manip/tasks/push_position/mdp/__init__.pyi",
    "src/stem_manip/tasks/push_position/mdp/events.py",
    "tests/test_fr3_reference.py",
    "tests/test_push_task_cfg.py",
]


def test_163_range_lists_the_push_stem_and_push_env_checks():
    checks = affected_checks.affected_checks(PATHS_163)
    assert "scripts/check_push_stem.py" in checks
    assert "scripts/check_push_env.py" in checks


def test_docs_only_change_lists_nothing():
    assert affected_checks.affected_checks(["docs/notes/2026-10-16.md", "README.md", "docs/adr/0004-slot-use.md"]) == []


def test_a_changed_check_script_lists_itself():
    assert affected_checks.affected_checks(["scripts/check_stem.py"]) == ["scripts/check_stem.py"]


def test_env_term_change_lists_every_push_check():
    checks = affected_checks.affected_checks(["src/stem_manip/tasks/push_position/mdp/actions.py"])
    assert checks == [
        "scripts/check_push_agents.py",
        "scripts/check_push_env.py",
        "scripts/check_push_obs.py",
        "scripts/check_push_stem.py",
        "scripts/check_push_terms.py",
    ]


def test_stem_asset_change_lists_the_stem_contact_and_push_checks():
    checks = affected_checks.affected_checks(["assets/stem/chain/chain.yaml"])
    assert "scripts/check_stem.py" in checks
    assert "scripts/check_contact.py" in checks
    assert "scripts/check_push_stem.py" in checks
    assert "scripts/check_fr3.py" not in checks


def test_ppo_agent_cfg_lists_nothing():
    # Used only by training; check_push_agents runs zero/random/scripted agents, not the PPO cfg.
    assert affected_checks.affected_checks(["src/stem_manip/tasks/push_position/agents/rsl_rl_ppo_cfg.py"]) == []


def test_unmapped_package_file_lists_every_check():
    # A shared helper with no finer rule: re-run everything rather than miss a check.
    checks = affected_checks.affected_checks(["src/stem_manip/utils/stem_geometry.py"])
    assert checks == sorted(affected_checks.ALL_CHECKS)
    assert len(checks) == 8
