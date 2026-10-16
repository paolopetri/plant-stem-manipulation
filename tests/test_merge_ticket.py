"""`scripts/merge_ticket.py` merges a ticket branch into an integration branch in one command (#180).

Each test builds a throwaway repo with a bare `origin`: integration branch `integ`, ticket branch `ticket`. The test
commands and the sim check runner are fake (python snippets), so no Kit and no GPU. The tests check:

- A dry run does every step and leaves the remote unchanged.
- It stops with a non-zero exit and names the cause on a conflict, a non-fast-forward push, a failing test and a
  failing sim check; nothing is pushed then.
- A change that touches no sim code runs no sim check; one that does runs its affected checks.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_ticket.py"
sys.path.insert(0, str(SCRIPT.parent))  # the script imports affected_checks from its own directory
spec = importlib.util.spec_from_file_location("merge_ticket", SCRIPT)
merge_ticket = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_ticket)

PASS = [sys.executable, "-c", "pass"]
FAIL = [sys.executable, "-c", "import sys; print('1 failed, 3 passed'); sys.exit(1)"]
# A fake sim check: records that it ran, passes unless the file has a `# FAILS` line.
CHECK = """import pathlib, sys
pathlib.Path("ran_" + pathlib.Path(__file__).stem).touch()
sys.exit("# FAIL" + "S" in open(__file__).read())
"""


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(cwd: Path, path: str, text: str) -> None:
    (cwd / path).parent.mkdir(parents=True, exist_ok=True)
    (cwd / path).write_text(text)
    git(cwd, "add", path)
    git(cwd, "commit", "-q", "-m", f"Change {path}")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clone on branch `ticket`, branched off `integ` on its origin; cwd is the clone; fake tests and runner."""
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "integ", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "integ", str(work)], check=True)
    git(work, "config", "user.email", "t@t")
    git(work, "config", "user.name", "t")
    commit(work, "README.md", "readme\n")
    commit(work, "scripts/check_fr3.py", CHECK)
    git(work, "remote", "add", "origin", str(origin))
    git(work, "push", "-q", "-u", "origin", "integ")
    git(work, "switch", "-q", "-c", "ticket")
    monkeypatch.chdir(work)
    monkeypatch.setattr(merge_ticket, "TESTS", [PASS])
    monkeypatch.setattr(merge_ticket, "SIM_CHECK", [sys.executable])
    monkeypatch.setattr(merge_ticket, "GH_COMMENT", [sys.executable, "-c", "import sys; print('commented', sys.argv)"])
    return work


def remote_tip(work: Path) -> str:
    return git(work, "ls-remote", "origin", "refs/heads/integ").split()[0]


def run(*args: str) -> int:
    return merge_ticket.main(["integ", "180", *args])


def test_dry_run_does_every_step_without_pushing(repo: Path, capsys: pytest.CaptureFixture):
    before = remote_tip(repo)
    commit(repo, "docs/x.md", "doc\n")
    assert run("--dry-run") == 0
    out = capsys.readouterr().out
    assert remote_tip(repo) == before
    assert "tests passed" in out
    assert "sim checks: none" in out
    assert "dry run: would push" in out
    assert "would comment on #180" in out
    assert "commented" not in out


def advance_integ(work: Path, path: str, text: str) -> None:
    """Another ticket lands on `integ` on origin."""
    git(work, "switch", "-q", "integ")
    commit(work, path, text)
    git(work, "push", "-q", "origin", "integ")
    git(work, "switch", "-q", "ticket")


def test_conflict_stops_nonzero_and_names_it(repo: Path, capsys: pytest.CaptureFixture):
    commit(repo, "README.md", "ticket side\n")
    advance_integ(repo, "README.md", "other ticket side\n")
    before = remote_tip(repo)
    assert run() != 0
    out = capsys.readouterr().out
    assert "FAILED (conflict)" in out and "README.md" in out
    assert remote_tip(repo) == before
    assert not (repo / ".git" / "MERGE_HEAD").exists()  # the merge was aborted


def test_branch_without_the_tip_gets_it_merged_in(repo: Path, capsys: pytest.CaptureFixture):
    commit(repo, "docs/a.md", "a\n")
    advance_integ(repo, "docs/b.md", "b\n")
    assert run() == 0
    assert "merged origin/integ" in capsys.readouterr().out
    assert remote_tip(repo) == git(repo, "rev-parse", "HEAD")
    assert (repo / "docs" / "b.md").exists()


def test_non_fast_forward_push_stops_nonzero(repo: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """The integration branch moves while the checks run: the push is rejected, never forced."""
    commit(repo, "docs/a.md", "a\n")
    clone = repo.parent / "other"
    subprocess.run(["git", "clone", "-q", "-b", "integ", str(repo.parent / "origin.git"), str(clone)], check=True)
    land = repo.parent / "land.py"  # the fake test step: another ticket lands on integ meanwhile
    land.write_text(
        f"import subprocess, uuid\nc = {str(clone)!r}\nopen(c + '/b.md', 'w').write(str(uuid.uuid4()))\n"
        "for a in (['add', '.'], ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'b'],"
        " ['push', '-q']):\n"
        "    subprocess.run(['git', '-C', c, *a], check=True)\n"
    )
    monkeypatch.setattr(merge_ticket, "TESTS", [[sys.executable, str(land)]])
    for args in (["--dry-run"], []):
        assert run(*args) != 0
        assert "FAILED (non-fast-forward)" in capsys.readouterr().out
    assert remote_tip(repo) == git(clone, "rev-parse", "HEAD")


def test_failing_test_stops_nonzero_and_shows_its_output(repo: Path, monkeypatch, capsys):
    commit(repo, "docs/a.md", "a\n")
    monkeypatch.setattr(merge_ticket, "TESTS", [PASS, FAIL])
    before = remote_tip(repo)
    assert run() != 0
    out = capsys.readouterr().out
    assert "FAILED (tests)" in out and "1 failed, 3 passed" in out
    assert remote_tip(repo) == before


def test_failing_sim_check_stops_nonzero_and_names_it(repo: Path, capsys: pytest.CaptureFixture):
    commit(repo, "scripts/check_fr3.py", CHECK + "# FAILS\n")
    before = remote_tip(repo)
    assert run() != 0
    out = capsys.readouterr().out
    assert "FAILED (sim check)" in out and "scripts/check_fr3.py" in out
    assert remote_tip(repo) == before


def test_sim_change_runs_its_affected_checks(repo: Path, capsys: pytest.CaptureFixture):
    commit(repo, "scripts/check_fr3.py", CHECK + "# changed\n")
    assert run() == 0
    assert "sim checks: scripts/check_fr3.py" in capsys.readouterr().out
    assert (repo / "ran_check_fr3").exists()


def test_change_without_sim_code_runs_no_sim_check(repo: Path, capsys: pytest.CaptureFixture):
    commit(repo, "docs/a.md", "a\n")
    commit(repo, "tests/test_a.py", "def test_a(): pass\n")
    assert run() == 0
    out = capsys.readouterr().out
    assert "sim checks: none" in out
    assert not (repo / "ran_check_fr3").exists()
    assert "commented" in out and "Merged into integ at" in out
