"""One cached object per env, e.g. the push task's slot-base kinematics (`mdp/slot_base.py`).

`per_env(build)` wraps `build(env)` so that it runs once per env; later calls return the same object. The objects
are kept in a module-level weak-key dictionary keyed by the env: nothing is written onto Isaac Lab's env object, and
an env that is garbage-collected drops its entry. The cached object must not hold a reference to the env (the
dictionary holds its values strongly, so the env would never be freed).
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable

_OBJECTS: weakref.WeakKeyDictionary[object, dict[Callable, object]] = weakref.WeakKeyDictionary()
"""env -> {build function: its object for that env}."""


def per_env[T](build: Callable[[object], T]) -> Callable[[object], T]:
    """Decorator: `build(env)` runs on the first call per env, later calls return its result."""

    @functools.wraps(build)
    def get(env: object) -> T:
        objects = _OBJECTS.setdefault(env, {})
        if build not in objects:
            objects[build] = build(env)
        return objects[build]

    return get
