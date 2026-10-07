"""Reset and randomization events.

Requirements:
- Reset robot and stem to their default states (stem: `reset_cable_state_uniform` or own term).
- Later: domain randomization of stem parameters (stiffness, length, diameter, damping) once the
  per-env randomization path is clarified. Ranges in the env cfg.

Implemented: base clamping at startup; the reset uses Isaac Lab's `reset_scene_to_default` (env cfg).
See docs/TODO.md -> M4 (randomization: Later).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from stem_manip.assets.stem import stem_model

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def fix_stem_base(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, model: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("stem")
) -> None:
    """Startup event: clamp the stem base with the stem model's `fix_stem_base` (`chain`: nothing to do)."""
    stem_model(model).fix_stem_base(env.scene[asset_cfg.name])
