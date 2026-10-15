"""Compare two rollout recordings of `scripts/record_rollout.py` (no simulator imports).

A recording is `{"meta": {...}, "terms": {name: tensor}}`: the settings it was recorded with and one tensor per
observation, reward and termination term, stacked over the policy steps. `compare` turns two recordings into
`check_report` rows: one per term (max absolute difference) and one for the settings. `never_nonzero` lets `record`
check that the push rollout still exercises the damage limits.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields

# Noise floor of the contact-sensor force terms (issue #150): their float32 contact solve differs by up to ~2.4e-7 N
# between recordings of unchanged code (another checkout, or a code change that leaves the physics alone), so they
# pass within it; every other term is bit-exact. Names match term keys as in `compare` (every rollout).
CONTACT_FORCE_TERMS = ("reward/contact_force", "state/robot_contact_force")
CONTACT_FORCE_NOISE = 1e-6


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
    a: Mapping, b: Mapping, tol: float | None = None, renames: Sequence[tuple[str, str]] = ()
) -> dict[str, tuple[bool, str]]:
    """Rows `{name: (ok, info)}`: the settings, then every term of either recording.

    Args:
        a, b: Recordings `{"meta": {...}, "terms": {name: tensor}}`.
        tol: Largest absolute difference that passes, for every term. None: bit-exact, except the
            `CONTACT_FORCE_TERMS`, which pass within `CONTACT_FORCE_NOISE` and are marked "float32 noise" when
            their difference is non-zero.
        renames: `(old, new)` for terms renamed between `a` and `b`: the terms of `a` are compared under their new
            names. A name matches a term key whole or as its tail after a `/` (`slot_base_pos` matches
            `zero/obs/slot_base_pos` in every rollout; `reward/curvature` only the reward term). The renames apply
            at once to the original names, so a swap works whatever their order.

    Raises:
        ValueError: An old name matches no term of `a` or a new name no term of `b`, an old name is given twice, a
            term of `a` is matched by two old names, or two terms of `a` get the same name: so no term drops out of
            the comparison unnoticed.
    """
    a_terms = _rename(a["terms"], b["terms"], renames)
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
            contact = tol is None and any(_matches(name, term) for term in CONTACT_FORCE_TERMS)
            ok = diff <= (CONTACT_FORCE_NOISE if contact else tol or 0.0)
            noise = " (float32 noise)" if contact and ok and diff > 0 else ""
            rows[name] = (ok, f"max |diff| {diff:.3g}{noise}")
    return rows


def _rename(a_terms: Mapping, b_terms: Mapping, renames: Sequence[tuple[str, str]]) -> dict:
    """The terms of `a` under their new names, checked as `compare` documents."""
    olds = [old for old, _ in renames]
    for old in olds:
        if olds.count(old) > 1:
            raise ValueError(f"{old!r} is renamed twice")
    for old, new in renames:
        for name, terms, which in ((old, a_terms, "first"), (new, b_terms, "second")):
            if not any(_matches(key, name) for key in terms):
                raise ValueError(f"{name!r} matches no term of the {which} recording (rename {old} -> {new})")
    renamed = {}
    for key, value in a_terms.items():
        hits = [(old, new) for old, new in renames if _matches(key, old)]
        if len(hits) > 1:
            raise ValueError(f"{key!r} is matched by both {hits[0][0]!r} and {hits[1][0]!r}")
        new_key = key[: -len(hits[0][0])] + hits[0][1] if hits else key
        if new_key in renamed:
            raise ValueError(f"{new_key!r} is the name of two terms of the first recording after the renames")
        renamed[new_key] = value
    return renamed


def _matches(key: str, name: str) -> bool:
    """Whether `name` is the term key `key` or its tail after a `/`."""
    return key == name or key.endswith("/" + name)


def never_nonzero(terms: Mapping, names: Sequence[str]) -> list[str]:
    """Those of `names` whose term is missing from `terms` or is 0 / False at every step and env, in order."""
    return [name for name in names if name not in terms or not terms[name].any()]
