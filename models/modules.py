from typing import Any

import torch
import torch.nn as nn

from loss.aggregators import LossOutput
from models.models.trm import TinyRecursiveModel
from utils.learning import LearningParameters

from .base import BaseLightningModule, LossAggregator


class MnistClassifierModule(BaseLightningModule):
    """
    Lightning module for the MNIST classifier, extends the BaseLightningModule class.
    """

    def __init__(
        self,
        model: TinyRecursiveModel,
        learning_params: LearningParameters,
        supervision_steps: int = 4,
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
        self.automatic_optimization = False
        self._supervision_steps = supervision_steps
        self.model = model

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the MNIST classifier.

        Args:
            x (dict[str, torch.Tensor]): Input data dictionary containing "images" tensor.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing "logits" tensor.
        """
        x = input["input"]
        y = input["inter output"]
        z = input["latent"]
        y, z, y_hat, q_hat = self.model.deep_recursion(x, y, z)
        return {"inter output": y, "latent": z, "output": y_hat, "stop": q_hat}

    def step(self, batch: dict[str, Any], phase: str) -> None:
        """
        Performs a single training/validation step.

        Args:
            batch (dict[str, Any]): Input batch data.
            phase (str): The phase of the training (e.g., "train", "val").

        Returns:
            torch.Tensor | None: The total loss if available, otherwise None.
        """
        y_init = torch.zeros((batch["images"].shape[0], 10)).to(batch["images"].device)
        z_init = torch.zeros((batch["images"].shape[0], 16)).to(batch["images"].device)

        y = y_init
        z = z_init

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        for _ in range(self._supervision_steps):
            sup_step_output = self.forward(
                {"input": batch["images"], "inter output": y, "latent": z}
            )
            if self.loss_aggregator is None:
                continue
            sup_step_output["logits"] = sup_step_output["output"]

            loss = self.loss_aggregator(sup_step_output, batch)
            self.log_loss(loss, phase)

            if phase != "training":
                continue

            self.manual_backward(loss.total)
            optimizer.step()

            if torch.all(sup_step_output["stop"] > 0):
                break

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
                sync_dist=True,
            )
        self.log(
            f"{phase} total loss",
            loss.total,
            prog_bar=True,
            batch_size=self.learning_params.batch_size,
            sync_dist=True,
        )
        return loss.total
