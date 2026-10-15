"""Stem metrics: a metrics-only term (a command term without a command) that logs how the policy loads the stem.

The metrics are logged in training (`Metrics/<term name>/<metric>`, mean over the envs that reset) and never observed
(no contact forces or stem loads in the observations, `docs/adr/0002-contact-forces-not-observed.md`). They are not
weight-0 rewards: they cost no reward computation and do not show up as rewards (spec #160).

- `max_compression` [N]: the episode's largest axial compression in any stem joint (`StemState.axial_compression`);
  0 if the stem was never compressed. Shows whether deep targets are reached by pressing the tip down (buckling,
  `docs/adr/0004-slot-use-through-target-choice.md`), and is the data for a compression limit (#82).
- `mean_compression` [N]: the mean over the episode's policy steps of the largest axial compression in any joint.
  At rest it is the stem's own weight above the lowest joint.

The first policy step after a reset is not counted: the joint forces are those of the last physics step, before the
reset wrote the new state.

Verify: `tests/test_stem_state.py` (the compression), a short sim run that prints the training log.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .stem_state import stem_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StemMetrics(CommandTerm):
    """Metrics of the stem, updated every policy step and logged at reset; its command is empty."""

    cfg: StemMetricsCfg

    def __init__(self, cfg: StemMetricsCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)
        self.metrics["max_compression"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["mean_compression"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """No command: shape (num_envs, 0)."""
        return self._empty

    def _resample_command(self, env_ids: Sequence[int]):
        pass  # no command; the metrics are zeroed by `CommandTerm.reset`

    def _update_command(self):
        pass

    def _update_metrics(self):
        # steps since the reset; 0 right after a reset, whose joint forces are still those before the reset
        steps = self._env.episode_length_buf
        counted = steps > 0
        compression = stem_state(self._env).axial_compression().amax(dim=-1)
        peak = self.metrics["max_compression"]
        mean = self.metrics["mean_compression"]
        peak[:] = torch.where(counted, torch.maximum(peak, compression), peak)
        mean[:] = torch.where(counted, mean + (compression - mean) / steps.clamp(min=1), mean)


@configclass
class StemMetricsCfg(CommandTermCfg):
    """Stem metrics (`StemMetrics`). Set `resampling_time_range` far above the episode length: nothing to resample."""

    class_type: type = StemMetrics
