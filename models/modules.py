from typing import Any

import torch
import torch.nn as nn

from loss.aggregators import LossOutput
from models.models import FCN
from utils.learning import LearningParameters

from .base import BaseLightningModule, LossAggregator


class MnistClassifierModule(BaseLightningModule):
    """
    Lightning module for the MNIST classifier, extends the BaseLightningModule class.
    """

    def __init__(
        self,
        model: FCN,
        learning_params: LearningParameters,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the Module class.

        Args:
            model (FCN): The model to be used.
            learning_params (LearningParameters): The learning parameters.
            transforms (nn.Sequential | None, optional): The data transforms. Defaults to None.
            loss_aggregator (LossAggregator | None, optional): The loss aggregator. Defaults to None.
            optimizer_cfg (dict[str, Any] | None, optional): The optimizer configuration. Defaults to None.
            scheduler_cfg (dict[str, Any] | None, optional): The scheduler configuration. Defaults to None.
        """
        super().__init__(
            model,
            learning_params,
            transforms,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the MNIST classifier.

        Args:
            x (dict[str, torch.Tensor]): Input data dictionary containing "images" tensor.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing "logits" tensor.
        """
        outputs = self.model(x["images"])
        return {"logits": outputs}

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        """
        Performs a single training/validation step.

        Args:
            batch (dict[str, Any]): Input batch data.
            phase (str): The phase of the training (e.g., "train", "val").

        Returns:
            torch.Tensor | None: The total loss if available, otherwise None.
        """
        outputs = self(batch)
        if self.loss_aggregator is not None:
            loss = self.loss_aggregator(outputs, batch)
            loss_total = self.log_loss(loss, phase)
            return loss_total

    def log_loss(self, loss: LossOutput, phase: str) -> torch.Tensor:
        """
        Handles the loss logging (to Tensorboard).

        Args:
            loss (LossOutput): The loss output object containing individual losses.
            phase (str): The phase of the training (e.g., "train", "val").

        Returns:
            torch.Tensor: The total loss.
        """
        for name in loss.individual:
            log_name = f"{phase} {name.replace('_', ' ')}"
            self.log(
                log_name, 
                loss.individual[name], 
                batch_size=self.learning_params.batch_size, 
                sync_dist=True
            )
        self.log(
            f"{phase} total loss", 
            loss.total, 
            prog_bar=True, 
            batch_size=self.learning_params.batch_size, 
            sync_dist=True
        )
        return loss.total
