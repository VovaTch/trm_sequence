from typing import Any, Generator

import torch
import torch.nn as nn

from loss.aggregators import LossOutput
from models.models.trm import TinyRecursiveModel
from utils.learning import LearningParameters
from utils.sample_schedulers.base import SampleScheduler

from .base import BaseLightningModule, LossAggregator


class LanguageTRMModule(BaseLightningModule):
    """
    Lightning module for the MNIST classifier, extends the BaseLightningModule class.
    """

    def __init__(
        self,
        model: TinyRecursiveModel,
        learning_params: LearningParameters,
        sample_scheduler: SampleScheduler,
        latent_len: int = 128,
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
        self._sample_scheduler = sample_scheduler
        self._latent_len = latent_len
        self._core_hidden_dim = model.core.hidden_dim

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the language TRM.

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
        y_init = torch.zeros(
            (batch["tokens"].shape[0], batch["tokens"].shape[1], self._core_hidden_dim)
        ).to(batch["tokens"].device)
        z_init = torch.zeros(
            (batch["tokens"].shape[0], self._latent_len, self._core_hidden_dim)
        ).to(batch["tokens"].device)

        y = y_init
        z = z_init

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        for _ in range(self._supervision_steps):
            sup_step_output = self.forward(
                {"input": batch["tokens"], "inter output": y, "latent": z}
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

    def generate(
        self,
        init_tokens: torch.Tensor | None = None,
        init_step: int = 0,
        seq_len: int = 1024,
        vocab_size: int = 66,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """
        Generates a tensor based on the provided initial latent tensor, initial step, and conditional tensor.

        Args:
            init_tokens (torch.Tensor, optional): The initial tokens to start the generation process.
                Defaults to None.
            init_step (int, optional): The initial step to start the generation process. Defaults to 0.
            seq_len (int, optional): The length of the sequence to generate. Defaults to 512.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 1024.
            temperature (float, optional): The temperature to use for sampling. Defaults to 0.7.

        Returns:
            torch.Tensor: The generated tensor.
        """
        raise NotImplementedError("TODO")

    def stream(
        self,
        init_tokens: torch.Tensor | None = None,
        init_step: int = 0,
        seq_len: int = 1024,
        vocab_size: int = 66,
        temperature: float = 0.7,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Streams a token tensor basted on the provided initial latent tensor and initial step.

        Args:
            init_tokens (torch.Tensor, optional): The initial tokens to start the generation process.
                Defaults to None.
            init_step (int, optional): The initial step to start the generation process. Defaults to 0.
            seq_len (int, optional): The length of the sequence to generate. Defaults to 1024.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 66.
            temperature (float, optional): The temperature to use for sampling. Defaults to 0.7.

        Returns:
            Generator[torch.Tensor, None, None]: A generator that yields the generated tensor at each step.
        """
        raise NotImplementedError("TODO")
