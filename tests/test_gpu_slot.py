"""`scripts/gpu_slot.py` runs a command once one of the two GPU slots is free (#151).

Three concurrent short commands on a private lock dir: the first fails after 1 s, the second runs 3 s.
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gpu_slot.py"


def _start(lock_dir: Path, name: str, seconds: float, status: int) -> subprocess.Popen:
    """Start a shell command through the script; it logs its start and end time to `<name>.start` / `<name>.end`."""
    log = lock_dir / name
    cmd = f"date +%s.%N > {log}.start; sleep {seconds}; date +%s.%N > {log}.end; exit {status}"
    return subprocess.Popen([sys.executable, str(SCRIPT), "--lock-dir", str(lock_dir), "sh", "-c", cmd])


def _wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} not written"
        time.sleep(0.05)


def _time(path: Path) -> float:
    return float(path.read_text())


def test_third_command_waits_for_a_free_slot(tmp_path: Path):
    failing = _start(tmp_path, "failing", 1.0, status=3)
    long = _start(tmp_path, "long", 3.0, status=0)
    _wait_for(tmp_path / "failing.start")
    _wait_for(tmp_path / "long.start")
    third = _start(tmp_path, "third", 0.1, status=0)

    assert failing.wait(timeout=10) == 3  # exit status passed through
    assert third.wait(timeout=10) == 0
    assert long.wait(timeout=10) == 0
    # the third starts in the slot the failing command freed, without waiting for the long one
    assert _time(tmp_path / "third.start") >= _time(tmp_path / "failing.end")
    assert _time(tmp_path / "third.start") < _time(tmp_path / "long.end")
