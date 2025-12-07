from typing import Any

import torch
import torch.nn as nn

from loss.aggregators import LossOutput
from utils.containers import LearningParameters
from .base import BaseLightningModule, LossAggregator


class AutoRegressorModule(BaseLightningModule):
    """
    Module designed to handle Mamba/Transformer model training.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_params: LearningParameters,
        val_interval: int = 16,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model,
            learning_params,
            transforms,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )
        self._val_count = 0
        self._val_interval = val_interval

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the autoregressor module.

        Args:
            input (dict[str, torch.Tensor]): Input dictionary containing the indices.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing the model predictions.
        """
        tokens = input["tokens"]
        outputs = self.model(tokens, tokens.shape[1])
        stacked_outputs = torch.stack(outputs, dim=-1)
        return {"logits": stacked_outputs}

    def handle_loss(self, loss: LossOutput, phase: str) -> torch.Tensor:
        """
        Handles the loss calculation and logging (to Tensorboard).

        Args:
            loss (LossOutput): The loss output object containing individual losses.
            phase (str): The phase of the training (e.g., "train", "val").

        Returns:
            torch.Tensor: The total loss.

        """
        for name in loss.individual:
            log_name = f"{phase}/{name.replace('_', ' ')}"
            self.log(log_name, loss.individual[name])
        self.log(f"{phase}/total loss", loss.total, prog_bar=True)
        return loss.total

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        """
        Performs a single step of the autoregressor module.

        Args:
            batch (dict[str, Any]): The input batch.
            phase (str): The phase of the training process.

        Returns:
            torch.Tensor | None: The total loss for the step, or None if the loss aggregator is not defined.
        """
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return None
        loss = self.loss_aggregator(output, batch)
        loss_total = self.handle_loss(loss, phase)
        return loss_total

    def on_validation_epoch_start(self) -> None:
        if self._val_count % self._val_interval == 0:
            pass  # TODO: generate text
