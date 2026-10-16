"""Scripted check phases start from the fixed start through one helper (#177; no simulator).
The tests check:

- `fixed_start` sets the start-pose event's params to the held-out pose a_above_tip alone, without joint noise, and
  leaves the other params alone.
- No `scripts/check_push_*.py` sets the joint noise to a constant itself (`joint_noise=0.0` and the like): a start
  without noise goes through `fixed_start`.
"""

import ast
from pathlib import Path

import pytest

from stem_manip.tasks.push_position.env_cfg import StemPushPositionEnvCfg, fixed_start

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CHECK_SCRIPTS = sorted(SCRIPTS.glob("check_push_*.py"))
A_ABOVE_TIP = (-0.013, -0.689, 0.011, -2.496, 0.007, 1.808, 0.779)  # [rad] held-out start pose (user, 2026-10-15)


def test_fixed_start_sets_a_above_tip_without_noise():
    params = StemPushPositionEnvCfg().events.start_pose.params
    asset_cfg = params["asset_cfg"]
    fixed_start(params)
    assert [tuple(p) for p in params["poses"]] == [A_ABOVE_TIP]
    assert params["joint_noise"] == 0.0
    assert params["asset_cfg"] is asset_cfg


def _constant_noise_writes(tree: ast.AST) -> list[int]:
    """Lines that set `joint_noise` to a constant: keyword argument, dict entry or subscript assignment."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "joint_noise" and isinstance(node.value, ast.Constant):
            lines.append(node.value.lineno)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "joint_noise" and isinstance(value, ast.Constant):
                    lines.append(value.lineno)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "joint_noise"
                ):
                    lines.append(node.lineno)
    return lines


def test_check_scripts_found():
    assert len(CHECK_SCRIPTS) >= 5


@pytest.mark.parametrize("script", CHECK_SCRIPTS, ids=lambda p: p.name)
def test_check_script_starts_without_noise_only_through_fixed_start(script: Path):
    lines = _constant_noise_writes(ast.parse(script.read_text()))
    assert not lines, f"{script.name} sets joint_noise to a constant at lines {lines}; use env_cfg.fixed_start"
