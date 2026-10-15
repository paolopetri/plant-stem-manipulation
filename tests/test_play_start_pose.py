"""A policy can be played from a chosen start pose (#168): `isaaclab play` passes Hydra overrides of the reset event's
`poses` (and `joint_noise`) through to the env cfg, and `scripts/start_poses.py pose` prints a working override line.
No simulator: the env cfg is resolved the way `isaaclab play` resolves it. Parsing Isaac Lab's launcher arguments
(play's parser, the script's) starts Kit's kernel, which needs OMNI_KIT_ACCEPT_EULA=YES: those tests are skipped
without it.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from isaaclab_tasks.utils.hydra import resolve_task_config

import stem_manip.tasks  # noqa: F401  (registers the task)

TASK = "StemManip-Push-Position-FR3-v0"
AGENT = "rsl_rl_cfg_entry_point"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start_poses.py"
POSE = (0.1, 0.2, 0.3, -2.0, 0.0, 2.0, 0.7)  # [rad] any pose, not in the table
OVERRIDES = [
    "env.events.start_pose.params.poses=[[" + ",".join(str(v) for v in POSE) + "]]",
    "env.events.start_pose.params.joint_noise=0.0",
]
# held-out start pose a_above_tip (user, 2026-10-15, #160): slot base at (0.40, 0, 0.50) m, as in test_fr3_reference
A_ABOVE_TIP = (-0.013, -0.689, 0.011, -2.496, 0.007, 1.808, 0.779)
needs_kit = pytest.mark.skipif(
    os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES",
    reason="parsing Isaac Lab's launcher arguments starts Kit's kernel, which needs OMNI_KIT_ACCEPT_EULA=YES",
)


def _start_pose_params(overrides: list[str]) -> dict:
    env_cfg, _ = resolve_task_config(TASK, AGENT, play_mode=True, overrides=overrides)
    return env_cfg.events.start_pose.params


def test_hydra_override_replaces_the_start_poses_and_the_noise():
    params = _start_pose_params(OVERRIDES)
    assert [list(p) for p in params["poses"]] == [list(POSE)]
    assert params["joint_noise"] == 0.0


def test_without_override_play_keeps_the_table():
    params = _start_pose_params([])
    assert len(params["poses"]) > 1
    assert list(POSE) not in [list(p) for p in params["poses"]]


@needs_kit
def test_play_command_line_passes_the_override(monkeypatch):
    """The arguments go through play's own parser (`play_rsl_rl._parse_args`), which hands the unknown ones to Hydra
    through `sys.argv`, then through play's config resolution."""
    play = pytest.importorskip("isaaclab_rl.entrypoints.backends.play_rsl_rl")
    monkeypatch.setattr(sys, "argv", ["isaaclab"])
    args = play._parse_args(["--task", TASK, "--num_envs", "2", *OVERRIDES])
    env_cfg, _ = resolve_task_config(args.task, args.agent, play_mode=not args.train_env_cfg)
    params = env_cfg.events.start_pose.params
    assert [list(p) for p in params["poses"]] == [list(POSE)]
    assert params["joint_noise"] == 0.0


@needs_kit
def test_start_poses_script_prints_a_working_override_for_a_held_out_pose():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "pose", "--held_out", "a_above_tip"],
        capture_output=True,
        text=True,
        check=True,
    )
    line = [ln for ln in result.stdout.splitlines() if "env.events.start_pose.params.poses=" in ln]
    assert len(line) == 1
    params = _start_pose_params(shlex.split(line[0]))
    assert [list(p) for p in params["poses"]] == [list(A_ABOVE_TIP)]
