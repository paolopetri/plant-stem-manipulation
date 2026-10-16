#!/usr/bin/env python3
"""Run a sim check through a GPU slot, log its full output and print only what an agent needs (standard library only).

Takes a GPU slot (`gpu_slot.py`), sets `OMNI_KIT_ACCEPT_EULA=YES` and runs the command. The full output goes to the
log file; the terminal gets only the slot lines, the `[PASS]` / `[FAIL]` / `[NOT RUN]` and verdict lines of
`stem_manip.utils.check_report`, any traceback and a final `exit N` line. The exit status is the command's.

Usage: scripts/sim_check.py uv run --extra isaacsim python scripts/check_fr3.py
(Run it with the system python, like `gpu_slot.py`.) Verify: tests/test_sim_check.py.
"""

import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from gpu_slot import LOCK_DIR, acquire_slot

STAGGER = 60.0  # s between two launches (shared shader cache), #174
LOG_DIR = Path("logs/sim_check")  # git-ignored

VERDICT = re.compile(r"^(\[(PASS|FAIL|NOT RUN)\] |=== (all passed|FAILED) ===$)")
TRACEBACK = re.compile(r"^(Traceback \(most recent call last\):|During handling of the above|The above exception was)")
SIM_LOG = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")  # Kit log lines start with a timestamp


class OutputFilter:
    """Pick the lines an agent needs: verdict lines and tracebacks, without the sim log lines inside a traceback."""

    def __init__(self) -> None:
        self.in_traceback = False

    def keep(self, line: str) -> bool:
        """True if `line` goes to the terminal; call it on every output line in order."""
        if TRACEBACK.match(line):
            self.in_traceback = True
            return True
        if self.in_traceback:
            if SIM_LOG.match(line) or not line.strip():
                return False
            if not line[0].isspace():  # the exception line ends the traceback
                self.in_traceback = False
            return True
        return bool(VERDICT.match(line))


def stagger_launch(lock_dir: Path, stagger: float) -> None:
    """Wait until `stagger` s have passed since the last launch of any runner, then record this launch."""
    with (lock_dir / "launch.lock").open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # one runner at a time reads and writes the last launch time
        f.seek(0)
        last = float(f.read() or 0.0)
        wait = last + stagger - time.time()
        if wait > 0:
            print(f"sim_check: staggering the launch, {wait:.0f} s", flush=True)
            time.sleep(wait)
        f.truncate(0)
        f.write(str(time.time()))


def _default_log(command: list[str]) -> Path:
    """`logs/sim_check/<time>-<check name>.log`, named after the first `.py` argument (else the command)."""
    name = next((Path(arg).stem for arg in command if arg.endswith(".py")), Path(command[0]).name)
    return LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.log"


def _run(command: list[str], log: Path, slot_fd: int) -> int:
    """Run the command, write every output line to `log`, print the ones that pass the filter; return its status.

    The command inherits the slot lock (`slot_fd`), so the slot stays taken while it runs, also if this runner dies.
    """
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    keep = OutputFilter().keep
    with log.open("w") as out:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, pass_fds=(slot_fd,)
        )
        for line in proc.stdout:
            out.write(line)
            if keep(line):
                print(line, end="", flush=True)
        status = proc.wait()
    return status if status >= 0 else 128 - status  # killed by a signal: the shell's convention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock-dir", type=Path, default=LOCK_DIR, help=f"GPU slot lock files (default {LOCK_DIR})")
    parser.add_argument("--stagger", type=float, default=STAGGER, help=f"s between launches (default {STAGGER})")
    parser.add_argument("--log", type=Path, help=f"log file (default {LOG_DIR}/<time>-<check>.log)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the command and its arguments")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")
    log = args.log or _default_log(command)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"sim_check: log {log}", flush=True)
    slot_fd = acquire_slot(args.lock_dir)
    stagger_launch(args.lock_dir, args.stagger)
    try:
        status = _run(command, log, slot_fd)
    except FileNotFoundError:
        print(f"sim_check: command not found: {command[0]}", flush=True)
        status = 127
    print(f"exit {status}", flush=True)
    sys.exit(status)


if __name__ == "__main__":
    main()
