"""Compare two rollout recordings of `scripts/record_rollout.py` (no simulator imports).

A recording is `{"meta": {...}, "terms": {name: tensor}}`: the settings it was recorded with and one tensor per
observation, reward and termination term, stacked over the policy steps. `compare` turns two recordings into
`check_report` rows: one per term (max absolute difference) and one for the settings. `never_nonzero` lets `record`
check that the push rollout still exercises the damage limits.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Settings:
    """Settings a recording was made with; two recordings are comparable only with equal settings."""

    task: str
    num_envs: int
    steps: int
    seed: int
    device: str
    push_steps: int
    """Policy steps of the push rollout (sum of the steps of its plan)."""


def meta(settings: Settings, commit: str) -> dict:
    """The `meta` entry of a recording: the settings and the commit (which differs on purpose: before / after)."""
    return asdict(settings) | {"commit": commit}


def compare(
    a: Mapping, b: Mapping, tol: float = 0.0, renames: Mapping[str, str] | None = None
) -> dict[str, tuple[bool, str]]:
    """Rows `{name: (ok, info)}`: the settings, then every term of either recording.

    Args:
        a, b: Recordings `{"meta": {...}, "terms": {name: tensor}}`.
        tol: Largest absolute difference that passes; 0 means bit-exact.
        renames: `{old: new}` for terms renamed between `a` and `b`: the terms of `a` are compared under their new
            names. A name matches a term key whole or as its tail after a `/` (`slot_base_pos` matches
            `zero/obs/slot_base_pos` in every rollout; `reward/curvature` only the reward term).

    Raises:
        ValueError: An old name matches no term of `a` or a new name no term of `b`, so a typo cannot hide a
            missing term.
    """
    a_terms = dict(a["terms"])
    for old, new in (renames or {}).items():
        for name, terms, which in ((old, a_terms, "the first"), (new, b["terms"], "the second")):
            if not any(_matches(key, name) for key in terms):
                raise ValueError(f"--rename {old}={new}: {name!r} matches no term of {which} recording")
        a_terms = {(key[: -len(old)] + new if _matches(key, old) else key): v for key, v in a_terms.items()}
    rows: dict[str, tuple[bool, str]] = {}
    keys = [f.name for f in fields(Settings)]
    diff_settings = [k for k in keys if a["meta"].get(k) != b["meta"].get(k)]
    rows["settings"] = (
        not diff_settings,
        ", ".join(f"{k} {a['meta'].get(k)} vs {b['meta'].get(k)}" for k in diff_settings)
        or f"commits {a['meta'].get('commit')} vs {b['meta'].get('commit')}",
    )
    names = list(a_terms) + [k for k in b["terms"] if k not in a_terms]
    for name in names:
        x, y = a_terms.get(name), b["terms"].get(name)
        if x is None or y is None:
            rows[name] = (False, f"only in {'the first' if y is None else 'the second'} recording")
        elif x.shape != y.shape:
            rows[name] = (False, f"shape {tuple(x.shape)} vs {tuple(y.shape)}")
        else:
            diff = float((x.double() - y.double()).abs().max()) if x.numel() else 0.0
            rows[name] = (diff <= tol, f"max |diff| {diff:.3g}")
    return rows


def _matches(key: str, name: str) -> bool:
    """Whether `name` is the term key `key` or its tail after a `/`."""
    return key == name or key.endswith("/" + name)


def never_nonzero(terms: Mapping, names: Sequence[str]) -> list[str]:
    """Those of `names` whose term is missing from `terms` or is 0 / False at every step and env, in order."""
    return [name for name in names if name not in terms or not terms[name].any()]
