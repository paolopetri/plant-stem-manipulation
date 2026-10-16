"""`scripts/sim_check.py` runs a sim check through a GPU slot and prints only what an agent needs.

Fake check commands (no Kit) on a private lock dir. The tests check:

- Only the slot, PASS/FAIL, traceback and `exit N` lines reach the output; the full output is in the log file;
  the exit status is the command's.
- A traceback is printed also after many timestamped sim warning lines, and with warning lines inside it.
- Two launches in a row are staggered by the runner itself.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sim_check.py"

WARNING = "2026-10-14T21:51:18Z [875ms] [Warning] [isaaclab.envs.manager_based_env] Seed not set for the environment."


def _run(tmp_path: Path, body: str, stagger: float = 0.0) -> tuple[subprocess.CompletedProcess, Path]:
    """Run a fake check (a python snippet) through the runner; return the result and the log path."""
    log = tmp_path / "check.log"
    check = tmp_path / "check_fake.py"
    check.write_text(body)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--lock-dir", str(tmp_path / "locks"), "--stagger", str(stagger)]
        + ["--log", str(log), sys.executable, str(check)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, log


def test_output_is_only_verdict_lines_and_exit_status(tmp_path: Path):
    body = f"""
import os, sys
print({WARNING!r})
print("Simulation App Startup Complete")
print("[INFO] stem tip height: 0.61 m")
print("[PASS] arm reaches start pose: 0.002 m")
print("[FAIL] stem tip deflects: 0.000 m")
print("[NOT RUN] stem springs back: no contact")
print("=== FAILED ===")
print("EULA", os.environ.get("OMNI_KIT_ACCEPT_EULA"), file=sys.stderr)
sys.exit(3)
"""
    result, log = _run(tmp_path, body)

    assert result.returncode == 3
    assert result.stdout.splitlines() == [
        f"sim_check: log {log}",
        "[PASS] arm reaches start pose: 0.002 m",
        "[FAIL] stem tip deflects: 0.000 m",
        "[NOT RUN] stem springs back: no contact",
        "=== FAILED ===",
        "exit 3",
    ]
    full = log.read_text()
    for line in (WARNING, "Simulation App Startup Complete", "[INFO] stem tip height: 0.61 m", "EULA YES"):
        assert line in full


def test_traceback_after_many_warning_lines_is_printed(tmp_path: Path):
    body = f"""
import sys
for _ in range(500):
    print({WARNING!r})
print("[PASS] arm reaches start pose: 0.002 m", flush=True)
import traceback
def step():
    raise RuntimeError("stem solver diverged")
try:
    step()
except RuntimeError:
    lines = traceback.format_exc().splitlines()
    # a sim warning between the traceback lines (Kit's log and stderr interleave)
    print("\\n".join(lines[:2] + [{WARNING!r}] + lines[2:]))
    sys.exit(1)
"""
    result, _ = _run(tmp_path, body)

    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert lines[1] == "[PASS] arm reaches start pose: 0.002 m"
    assert lines[2] == "Traceback (most recent call last):"
    assert lines[-2] == "RuntimeError: stem solver diverged"
    assert lines[-1] == "exit 1"
    assert not any(line.startswith("2026-") for line in lines)
    assert any("step()" in line for line in lines)  # the frames are printed


def test_two_launches_in_a_row_are_staggered(tmp_path: Path):
    body = f"import time\nopen({str(tmp_path / 'starts')!r}, 'a').write(f'{{time.time()}}\\n')\n"
    first, _ = _run(tmp_path, body, stagger=2.0)
    second, _ = _run(tmp_path, body, stagger=2.0)

    assert first.returncode == second.returncode == 0
    start_first, start_second = (float(t) for t in (tmp_path / "starts").read_text().split())
    assert start_second - start_first >= 2.0
