from typing import Any

import torch
import torch.nn as nn

from loss.aggregators import LossAggregator
from models.models.trm import TinyRecursiveModel
from models.modules.trm_diffusion import LanguageTRMModule
from utils.containers import LearningParameters
from utils.sample_schedulers.base import SampleScheduler


class MuonLanguageTRMModule(LanguageTRMModule):
    def __init__(
        self,
        model: TinyRecursiveModel,
        learning_params: LearningParameters,
        sample_scheduler: SampleScheduler,
        latent_len: int = 128,
        supervision_steps: int = 4,
        gradient_clip: float = 0.1,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model,
            learning_params,
            sample_scheduler,
            latent_len,
            supervision_steps,
            gradient_clip,
            transforms,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )
        self.optimizers = 0  # type: ignore

        core_parameters = list(self.model.core.parameters())
        self._muon_optimizer = torch.optim.Muon()
