#!/usr/bin/env python3
"""Run a command once one of the GPU slots is free (no simulator, standard library only).

Isaac Sim runs of parallel sessions (checks, recordings) share the GPU; each one takes a slot. A slot is an `flock`
on a lock file in `--lock-dir`. The script waits for a slot without polling (blocking `flock`), then replaces itself
with the command (`exec`), which keeps the lock: the slot is freed when the command ends, also on a crash or kill,
and the command's exit status is the script's.

Usage: scripts/gpu_slot.py uv run --extra isaacsim python scripts/check_fr3.py
(Run it with the system python, not `uv run`: a plain `uv run` would sync the .venv without the isaacsim extra.)
Verify: tests/test_gpu_slot.py.
"""

import argparse
import fcntl
import os
import queue
import sys
import threading
from pathlib import Path

SLOTS = 2  # Isaac Sim runs on the GPU at the same time (#47)
LOCK_DIR = Path("/tmp/stem-manip-gpu-slots")  # shared by all checkouts and worktrees


def acquire_slot(lock_dir: Path) -> int:
    """Block until a slot is free and return the descriptor that holds its lock."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    fds = [os.open(lock_dir / f"slot{i}.lock", os.O_RDWR | os.O_CREAT, 0o666) for i in range(SLOTS)]
    for fd in fds:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            pass
    print(f"gpu_slot: all {SLOTS} GPU slots busy, waiting", file=sys.stderr, flush=True)
    # one blocking flock per slot; the first one to get its lock wins. A later winner holds its lock only until the
    # exec, which closes every descriptor but the returned one.
    locked: queue.Queue[int] = queue.Queue()

    def wait(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked.put(fd)

    for fd in fds:
        threading.Thread(target=wait, args=(fd,), daemon=True).start()
    fd = locked.get()
    print("gpu_slot: got a GPU slot", file=sys.stderr, flush=True)
    return fd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock-dir", type=Path, default=LOCK_DIR, help=f"lock files (default {LOCK_DIR})")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the command and its arguments")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")
    fd = acquire_slot(args.lock_dir)
    os.set_inheritable(fd, True)  # the command keeps the lock; the other descriptors close on exec
    try:
        os.execvp(command[0], command)
    except FileNotFoundError:
        print(f"gpu_slot: command not found: {command[0]}", file=sys.stderr)
        sys.exit(127)


if __name__ == "__main__":
    main()
