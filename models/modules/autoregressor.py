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
            self.log(log_name, loss.individual[name], sync_dist=True)
        self.log(f"{phase}/total loss", loss.total, prog_bar=True, sync_dist=True)
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

    def generate_next_tokens(
        self, input_seq: torch.Tensor, temperature: float = 0.7, top_k: int = 0
    ) -> torch.Tensor:
        if temperature < 0:
            raise ValueError(f"Temperature must be non negative, got {temperature}")
        outputs = self.model(input_seq, 1)  # Expected BS x 1 x C
        if top_k > 0:
            values, _ = torch.topk(outputs, min(top_k, outputs.shape[-1]), dim=2)
            kth_value = values[:, -1].unsqueeze(-1)
            outputs = torch.where(outputs < kth_value, -float("inf"), outputs)
        probs = torch.softmax(outputs / (temperature + 1e-8), dim=2)
        dist = torch.distributions.Categorical(probs)
        sampled_tokens = dist.sample()
        return sampled_tokens
