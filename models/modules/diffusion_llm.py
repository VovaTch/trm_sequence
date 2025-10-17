from typing import Any, Generator

import torch
import torch.nn as nn
import tqdm

from loss.aggregators import LossOutput
from utils.containers import LearningParameters
from utils.sample_schedulers.base import SampleScheduler
from .base import BaseLightningModule, LossAggregator


class DiffusionLLMLightningModule(BaseLightningModule):
    """
    A module to perform training using the method presented in LLaDa paper.
    https://arxiv.org/pdf/2502.09992
    """

    def __init__(
        self,
        model: nn.Module,
        learning_params: LearningParameters,
        sample_scheduler: SampleScheduler,
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
        self._sample_scheduler = sample_scheduler

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.model(input["tokens"], input["mask"])
        return {"logits": outputs}

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return
        loss = self.loss_aggregator(output, batch)
        loss_total = self.handle_loss(loss, phase)
        return loss_total

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
            log_name = f"{phase} {name.replace('_', ' ')}"
            self.log(
                log_name,
                loss.individual[name],
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )
        self.log(
            f"{phase} total loss",
            loss.total,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )
        return loss.total

    @torch.no_grad()
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
        if temperature < 0:
            raise ValueError("Temperature must be non-negative.")

        num_steps = self._sample_scheduler.num_steps
        current_tokens = (
            torch.cat(
                (
                    init_tokens.to(self._device),
                    torch.zeros(
                        (1, seq_len - len(init_tokens)),
                        device=self._device,
                        dtype=torch.int64,
                    ),
                ),
                dim=1,
            )
            if init_tokens is not None
            else torch.zeros((1, seq_len), device=self._device, dtype=torch.int64)
        )
        current_logits = torch.randn((1, seq_len, vocab_size), device=self._device)
        current_mask = torch.zeros_like(
            current_tokens, dtype=torch.bool, device=self._device
        )
        current_mask[:, :init_step] = True
        for step in tqdm.tqdm(range(num_steps), desc="Generating diffusion tokens..."):
            current_mask = (
                self._sample_scheduler.sample(step, current_mask, current_logits)
                .to(dtype=torch.bool)
                .to(self._device)
            )
            if step > init_step:
                current_logits = self.model(current_tokens, current_mask)
                cat_probs = torch.softmax(current_logits, dim=-1)
                cat_distribution = torch.distributions.Categorical(
                    cat_probs ** (1 / (temperature + 1e-9))
                )
                sampled_latent = cat_distribution.sample()
                current_tokens[~current_mask] = sampled_latent[~current_mask]
        return current_tokens

    @torch.no_grad()
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
        if temperature < 0:
            raise ValueError("Temperature must be non-negative.")

        num_steps = self._sample_scheduler.num_steps
        current_tokens = (
            torch.cat(
                (
                    init_tokens.to(self._device),
                    torch.zeros(
                        (1, seq_len - len(init_tokens)),
                        device=self._device,
                        dtype=torch.int64,
                    ),
                ),
                dim=1,
            )
            if init_tokens is not None
            else torch.zeros((1, seq_len), device=self._device, dtype=torch.int64)
        )
        current_logits = torch.randn((1, seq_len, vocab_size), device=self._device)
        current_mask = torch.zeros_like(
            current_tokens, dtype=torch.bool, device=self._device
        )
        current_mask[:, :init_step] = True
        for step in range(num_steps):
            current_mask = (
                self._sample_scheduler.sample(step, current_mask, current_logits)
                .to(dtype=torch.bool)
                .to(self._device)
            )
            if step > init_step:
                current_logits = self.model(current_tokens, current_mask)
                cat_probs = torch.softmax(current_logits, dim=-1)
                cat_distribution = torch.distributions.Categorical(
                    cat_probs ** (1 / (temperature + 1e-9))
                )
                sampled_latent = cat_distribution.sample()
                current_tokens[~current_mask] = sampled_latent[~current_mask]
            yield current_tokens
