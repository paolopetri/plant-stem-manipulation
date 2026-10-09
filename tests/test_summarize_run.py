"""`scripts/summarize_run.py` turns a run's tensorboard events into a compact table (no simulator).
Synthetic event file with the tag names RSL-RL 5.5 and Isaac Lab's managers write (M5 plan, step 4).
"""

import importlib.util
import math
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_run.py"
spec = importlib.util.spec_from_file_location("summarize_run", SCRIPT)
summarize_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_run)

ITERATIONS = 100  # logged at 0 ... 99


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    writer = SummaryWriter(log_dir=str(tmp_path))
    for it in range(ITERATIONS):
        writer.add_scalar("Train/mean_reward", float(it), it)
        writer.add_scalar("Train/mean_reward/time", float(it), 10 * it)  # wall-time duplicate: left out
        writer.add_scalar("Episode_Reward/distance_fine", 0.01 * it, it)
        writer.add_scalar("Episode_Termination/joint_margin", 0.5, it)
        writer.add_scalar("Metrics/stem_target/position_error", 0.1 - 0.001 * it, it)
        writer.add_scalar("Loss/value_function", math.nan if it == 70 else 1.0, it)
        writer.add_scalar("Perf/total_fps", 5000.0, it)
    writer.close()
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "env.yaml").write_text("decimation: 16\ngit_commit: e7c48d761d31\nseed: 42\n")
    return tmp_path


def test_values_at_evenly_spaced_iterations(run_dir: Path):
    scalars = summarize_run.load_scalars(run_dir)
    assert summarize_run.pick_iterations(scalars, points=4) == [0, 33, 66, 99]
    row = next(line for line in summarize_run.summarize(run_dir, points=4).splitlines() if "Train/mean_reward" in line)
    assert row.split()[1:5] == ["0", "33", "66", "99"]


def test_tag_order_and_wall_time_tags_left_out(run_dir: Path):
    text = summarize_run.summarize(run_dir, points=4)
    order = ["Train/mean_reward", "Episode_Reward/", "Episode_Termination/", "Metrics/", "Loss/", "Perf/total_fps"]
    positions = [text.index(tag) for tag in order]
    assert positions == sorted(positions)
    assert "/time" not in text


def test_non_finite_values_are_flagged(run_dir: Path):
    assert "NON-FINITE: Loss/value_function at iteration 70" in summarize_run.summarize(run_dir, points=4)


def test_header_has_the_commit_and_the_last_iteration(run_dir: Path):
    header = summarize_run.summarize(run_dir, points=4).splitlines()[:3]
    assert any("git_commit e7c48d761d31" in line for line in header)
    assert any("last iteration 99" in line for line in header)
