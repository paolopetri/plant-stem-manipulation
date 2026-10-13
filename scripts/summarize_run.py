"""Compact summary of a training run from its local tensorboard events (no simulator).

Prints the run's commit (`params/env.yaml`), then one row per logged scalar with its values at `--points` evenly
spaced iterations: mean reward and episode length, every reward term (`Episode_Reward/*`, per second of episode),
every termination's share (`Episode_Termination/*`), the command metrics (`Metrics/*`, e.g. the stem tip's
`position_error` at the end of the episode), noise std, losses, fps. Non-finite values are flagged. Meant to be read
by people and by the weekend training agents (keeps their input short; M5 plan, step 4).

Usage: uv run python scripts/summarize_run.py logs/rsl_rl/stem_push_position/<run> [--points 10]
Verify: tests/test_summarize_run.py.
"""

import argparse
import math
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# row order: tags starting with these prefixes, in this order; anything else at the end
TAG_ORDER = ["Train/", "Episode_Reward/", "Episode_Termination/", "Metrics/", "Policy/", "Loss/", "Perf/total_fps"]
LEFT_OUT = re.compile(r"/time$|^Perf/(collection|learning)_time$")  # wall-time duplicates, timing details


def load_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
    """All scalars of the run's event files: tag -> {iteration: value} (later files win, e.g. after a resume)."""
    scalars: dict[str, dict[int, float]] = {}
    for path in sorted(Path(run_dir).glob("events.out.tfevents.*")):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        for tag in events.Tags()["scalars"]:
            if not LEFT_OUT.search(tag):
                scalars.setdefault(tag, {}).update({e.step: e.value for e in events.Scalars(tag)})
    return scalars


def pick_iterations(scalars: dict[str, dict[int, float]], points: int) -> list[int]:
    """`points` evenly spaced iterations from the first to the last logged one (nearest logged iteration)."""
    logged = sorted(set().union(*scalars.values()))
    first, last = logged[0], logged[-1]
    wanted = [first + (last - first) * i / max(points - 1, 1) for i in range(points)]
    return sorted({min(logged, key=lambda it, w=w: abs(it - w)) for w in wanted})


def _order(tag: str) -> tuple[int, str]:
    rank = next((i for i, prefix in enumerate(TAG_ORDER) if tag.startswith(prefix)), len(TAG_ORDER))
    return rank, tag


def _commit(run_dir: Path) -> str:
    env_yaml = Path(run_dir) / "params" / "env.yaml"
    match = env_yaml.is_file() and re.search(r"^git_commit: *(\S+)", env_yaml.read_text(), re.MULTILINE)
    return match.group(1) if match else "unknown"


def summarize(run_dir: Path, points: int = 10) -> str:
    """The summary as text: header, one row per tag, non-finite flags."""
    scalars = load_scalars(run_dir)
    if not scalars:
        return f"run {run_dir}: no scalars logged yet"
    iterations = pick_iterations(scalars, points)
    logged = set().union(*scalars.values())
    width = max(len(tag) for tag in scalars)
    lines = [
        f"run {run_dir}",
        f"git_commit {_commit(run_dir)} | last iteration {max(logged)} | {len(logged)} iterations logged",
        f"{'iteration':<{width}} " + " ".join(f"{it:>9}" for it in iterations),
    ]
    flags = []
    for tag in sorted(scalars, key=_order):
        values = scalars[tag]
        cells = [f"{values[it]:>9.4g}" if it in values else f"{'-':>9}" for it in iterations]
        lines.append(f"{tag:<{width}} " + " ".join(cells))
        bad = [it for it, value in sorted(values.items()) if not math.isfinite(value)]
        if bad:
            flags.append(f"NON-FINITE: {tag} at iteration {bad[0]} ({len(bad)} iterations)")
    return "\n".join(lines + flags)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="run directory with the tensorboard events")
    parser.add_argument("--points", type=int, default=10, help="number of iterations shown per row")
    args = parser.parse_args()
    print(summarize(args.run_dir, args.points))


if __name__ == "__main__":
    main()
