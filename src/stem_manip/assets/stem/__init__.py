"""Stem models, selectable by name (like the end-effectors).

A stem model `<name>` consists of its values in `assets/stem/<name>/<name>.yaml` and the module
`stem_manip.assets.stem.<name>`. The plant values shared by all models are in `assets/stem/stem.yaml`.
Each model module provides:
- `stem_cfg()`: the Isaac Lab asset cfg of the upright stem (set `prim_path` in the scene);
- `physics_cfg()`: the physics cfg of the scene (the engine follows from the model);
- `fix_stem_base(stem)`: clamp the base after the simulation is built.

Models: `cable` (Newton cable, VBD solver).
"""

import importlib
from types import ModuleType

import yaml

from stem_manip.assets import REPO_ASSETS_DIR

STEM_DIR = REPO_ASSETS_DIR / "stem"
STEM_MODELS = sorted(path.name for path in STEM_DIR.iterdir() if (path / f"{path.name}.yaml").is_file())
"""Names of the stem models (folders `assets/stem/<name>/` with a `<name>.yaml`)."""


def stem_params(model: str) -> dict:
    """Stem parameters of `model`: the plant values of `stem.yaml` merged with `<model>/<model>.yaml`.

    Both files use the same sections (geometry, material, ...); a key may be defined in only one of them.
    """
    params = yaml.safe_load((STEM_DIR / "stem.yaml").read_text())
    for section, values in yaml.safe_load((STEM_DIR / model / f"{model}.yaml").read_text()).items():
        shared = params.setdefault(section, {})
        duplicates = shared.keys() & values.keys()
        if duplicates:
            raise ValueError(f"{model}.yaml redefines plant values from stem.yaml: {section}.{sorted(duplicates)}")
        shared.update(values)
    return params


def stem_model(name: str) -> ModuleType:
    """The module of stem model `name`, imported on first use (so only the chosen model's engine is loaded)."""
    if name not in STEM_MODELS:
        raise ValueError(f"Unknown stem model '{name}', available: {STEM_MODELS}")
    return importlib.import_module(f"{__name__}.{name}")
