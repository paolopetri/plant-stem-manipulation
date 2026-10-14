"""Compare two rollout recordings of `scripts/record_rollout.py` (no simulator imports).

A recording is `{"meta": {...}, "terms": {name: tensor}}`: the settings it was recorded with and one tensor per
observation, reward and termination term, stacked over the policy steps. `compare` turns two recordings into
`check_report` rows: one per term (max absolute difference) and one for the settings.
"""

from collections.abc import Mapping

# settings that must match for two recordings to be comparable; the commit differs on purpose (before / after)
SETTINGS = ("task", "num_envs", "steps", "seed", "device")


def compare(a: Mapping, b: Mapping, tol: float = 0.0) -> dict[str, tuple[bool, str]]:
    """Rows `{name: (ok, info)}`: the settings, then every term of either recording.

    Args:
        a, b: Recordings `{"meta": {...}, "terms": {name: tensor}}`.
        tol: Largest absolute difference that passes; 0 means bit-exact.
    """
    rows: dict[str, tuple[bool, str]] = {}
    diff_settings = [k for k in SETTINGS if a["meta"].get(k) != b["meta"].get(k)]
    rows["settings"] = (
        not diff_settings,
        ", ".join(f"{k} {a['meta'].get(k)} vs {b['meta'].get(k)}" for k in diff_settings)
        or f"commits {a['meta'].get('commit')} vs {b['meta'].get('commit')}",
    )
    names = list(a["terms"]) + [k for k in b["terms"] if k not in a["terms"]]
    for name in names:
        x, y = a["terms"].get(name), b["terms"].get(name)
        if x is None or y is None:
            rows[name] = (False, f"only in {'the first' if y is None else 'the second'} recording")
        elif x.shape != y.shape:
            rows[name] = (False, f"shape {tuple(x.shape)} vs {tuple(y.shape)}")
        else:
            diff = float((x.double() - y.double()).abs().max()) if x.numel() else 0.0
            rows[name] = (diff <= tol, f"max |diff| {diff:.3g}")
    return rows
