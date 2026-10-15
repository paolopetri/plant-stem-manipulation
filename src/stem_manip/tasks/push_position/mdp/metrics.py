"""Stem metrics: a metrics-only term (a command term without a command) that logs how the policy loads the stem.

The metrics are logged in training (`Metrics/<term name>/<metric>`, mean over the envs that reset) and never observed
(no contact forces or stem loads in the observations, `docs/adr/0002-contact-forces-not-observed.md`). They are not
weight-0 rewards: they cost no reward computation and do not show up as rewards (spec #160).

- `max_compression` [N]: the episode's largest axial compression in any stem joint (`StemState.axial_compression`);
  0 if the stem was never compressed. Shows whether deep targets are reached by pressing the tip down (buckling,
  `docs/adr/0004-slot-use-through-target-choice.md`), and is the data for a compression limit (#82).
- `mean_compression` [N]: the mean over the episode's policy steps of the largest axial compression in any joint.
  At rest it is the stem's own weight above the lowest joint.
- `arm_contact` [share of episodes]: 1 where the episode had an arm contact (the contact force of a robot body other
  than the fork above `arm_contact_force`, `contact.arm_contact_force`), else 0. Logged as the share of the episodes
  that ended in the policy step, next to `Episode_Termination/arm_contact` (the share that ended by it); it also
  counts arm contact while the termination is off. The policy step that ends an episode reaches the metrics only
  after the reset: an arm contact there is taken from the termination term `arm_contact`, where active.

The first policy step after a reset is not counted for the compression: the joint forces are those of the last
physics step, before the reset wrote the new state.

Verify: `tests/test_stem_state.py` (the compression), `scripts/check_push_terms.py` (arm contact), a short sim run
that prints the training log.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .contact import arm_contact_force
from .stem_state import stem_state

ARM_CONTACT_TERM = "arm_contact"  # the termination term whose last value covers the policy step that ends an episode

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
        self.metrics["arm_contact"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """No command: shape (num_envs, 0)."""
        return self._empty

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Add the arm contact of the policy step that ended the episode (`ARM_CONTACT_TERM`), then log and zero the
        metrics. Only where the episode has policy steps: a reset right after a reset adds nothing."""
        terminations = self._env.termination_manager
        if ARM_CONTACT_TERM in terminations.active_terms:
            ids = slice(None) if env_ids is None else env_ids
            ended = terminations.get_term(ARM_CONTACT_TERM)[ids] & (self._env.episode_length_buf[ids] > 0)
            self.metrics["arm_contact"][ids] = torch.maximum(self.metrics["arm_contact"][ids], ended.float())
        return super().reset(env_ids)

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
        arm = arm_contact_force(self._env, self.cfg.sensor_cfg) > self.cfg.arm_contact_force
        self.metrics["arm_contact"][:] = torch.maximum(self.metrics["arm_contact"], arm.float())


@configclass
class StemMetricsCfg(CommandTermCfg):
    """Stem metrics (`StemMetrics`). Set `resampling_time_range` far above the episode length: nothing to resample."""

    class_type: type = StemMetrics
    sensor_cfg: SceneEntityCfg = MISSING
    """The stem contact sensor (`contact.arm_contact_force`)."""
    arm_contact_force: float = MISSING
    """[N] arm contact above this force (the threshold of the termination `arm_contact`)."""
