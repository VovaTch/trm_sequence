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
        gradient_clip: float = 0.1,
        random_step_mask: bool = True,
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
        self._gradient_clip = gradient_clip
        self._random_step_mask = random_step_mask

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
        y_init = self.model.core.y_init.repeat(
            (batch["tokens"].shape[0], batch["tokens"].shape[1], 1)
        ).to(batch["tokens"].device)
        z_init = self.model.core.z_init.repeat(
            (batch["tokens"].shape[0], self._latent_len, 1)
        ).to(batch["tokens"].device)

        y = y_init
        z = z_init

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        total_loss_output = LossOutput(torch.zeros(1).to(batch["tokens"].device), {})

        for idx in range(self._supervision_steps):

            masked_token_input = batch["tokens"].clone()
            if self._random_step_mask:
                ber_prob = (
                    torch.ones_like(batch["tokens"]) * idx / self._supervision_steps
                )
                ber_dist = torch.distributions.Bernoulli(ber_prob)
                batch["mask"] = ber_dist.sample().to(dtype=torch.bool)
            masked_token_input[~batch["mask"]] = self.model.core.vocab_size

            sup_step_output = self.forward(
                {"input": masked_token_input, "inter output": y, "latent": z}
            )
            if self.loss_aggregator is None:
                continue
            sup_step_output["logits"] = sup_step_output["output"]
            y = sup_step_output["inter output"]
            z = sup_step_output["latent"]

            loss = self.loss_aggregator(sup_step_output, batch)
            total_loss_output.total += loss.total.detach() / self._supervision_steps
            for name in loss.individual:
                total_loss_output.individual[name] = (
                    total_loss_output.individual.get(name, 0)
                    + loss.individual[name].detach() / self._supervision_steps
                )

            if phase != "training":
                continue

            self.manual_backward(loss.total)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._gradient_clip)
            optimizer.step()
            optimizer.zero_grad()

            if torch.all(sup_step_output["stop"] > 0):
                break

        self.log_loss(total_loss_output, phase)

        if phase != "training":
            return
        scheduler = self.lr_schedulers()
        if isinstance(scheduler, list):
            scheduler = scheduler[0]
        if scheduler is not None:
            scheduler.step()  # type: ignore

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
        vocab_size: int = 65,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """
        Generates a tensor based on the provided initial latent tensor, initial step, and conditional tensor.

        Args:
            init_tokens (torch.Tensor, optional): The initial tokens to start the generation process.
                Defaults to None.
            init_step (int, optional): The initial step to start the generation process. Defaults to 0.
            seq_len (int, optional): The length of the sequence to generate. Defaults to 1024.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 65.
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
                    torch.randint(
                        0,
                        self.model.core.vocab_size,
                        (1, seq_len - init_tokens.shape[-1]),
                        device=self._device,
                        dtype=torch.int64,
                    ),
                ),
                dim=1,
            )
            if init_tokens is not None
            else torch.randint(
                0,
                self.model.core.vocab_size,
                (1, seq_len),
                device=self._device,
                dtype=torch.int64,
            )
        )
        current_logits = torch.randn((1, seq_len, vocab_size), device=self._device)
        current_mask = torch.zeros_like(
            current_tokens, dtype=torch.bool, device=self._device
        )
        if init_tokens is None:
            init_token_len = 0
        else:
            init_token_len = init_tokens.shape[-1]
        current_mask[:, :init_token_len] = True

        y_init = self.model.core.y_init.repeat((1, seq_len, 1)).to(self._device)
        z_init = self.model.core.z_init.repeat((1, self._latent_len, 1)).to(
            self._device
        )

        y = y_init
        z = z_init

        for step in range(num_steps):
            current_mask = (
                self._sample_scheduler.sample(step, current_mask, current_logits)
                .to(dtype=torch.bool)
                .to(self._device)
            )
            current_mask[:, :init_token_len] = True
            if step >= init_step:
                current_tokens[~current_mask] = self.model.core.vocab_size
                step_output = self.forward(
                    {"input": current_tokens, "inter output": y, "latent": z}
                )
                current_logits = step_output["output"]
                cat_probs = torch.softmax(current_logits, dim=-1)
                cat_distribution = torch.distributions.Categorical(
                    cat_probs ** (1 / (temperature + 1e-9))
                )
                sampled_latent = cat_distribution.sample()
                current_tokens[~current_mask] = sampled_latent[~current_mask]

                y = step_output["inter output"]
                z = step_output["latent"]

                if torch.all(step_output["stop"] > 0):
                    break

        return current_tokens

    @torch.no_grad()
    def stream(
        self,
        init_tokens: torch.Tensor | None = None,
        init_step: int = 0,
        seq_len: int = 1024,
        vocab_size: int = 65,
        temperature: float = 0.7,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Streams a token tensor basted on the provided initial latent tensor and initial step.

        Args:
            init_tokens (torch.Tensor, optional): The initial tokens to start the generation process.
                Defaults to None.
            init_step (int, optional): The initial step to start the generation process. Defaults to 0.
            seq_len (int, optional): The length of the sequence to generate. Defaults to 1024.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 65.
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
                    torch.randint(
                        0,
                        self.model.core.vocab_size,
                        (1, seq_len - init_tokens.shape[-1]),
                        device=self._device,
                        dtype=torch.int64,
                    ),
                ),
                dim=1,
            )
            if init_tokens is not None
            else torch.randint(
                0,
                self.model.core.vocab_size,
                (1, seq_len),
                device=self._device,
                dtype=torch.int64,
            )
        )

        current_logits = torch.randn((1, seq_len, vocab_size), device=self._device)
        current_mask = torch.zeros_like(
            current_tokens, dtype=torch.bool, device=self._device
        )

        if init_tokens is None:
            init_token_len = 0
        else:
            init_token_len = init_tokens.shape[-1]
        current_mask[:, :init_token_len] = True

        y_init = self.model.core.y_init.repeat((1, seq_len, 1)).to(self._device)
        z_init = self.model.core.z_init.repeat((1, self._latent_len, 1)).to(
            self._device
        )

        y = y_init
        z = z_init

        step_output = None

        for step in range(num_steps):
            current_mask = (
                self._sample_scheduler.sample(step, current_mask, current_logits)
                .to(dtype=torch.bool)
                .to(self._device)
            )
            current_mask[:, :init_token_len] = True
            if step >= init_step:
                current_tokens[~current_mask] = self.model.core.vocab_size
                step_output = self.forward(
                    {"input": current_tokens, "inter output": y, "latent": z}
                )
                current_logits = step_output["output"]
                cat_probs = torch.softmax(current_logits, dim=-1)
                cat_distribution = torch.distributions.Categorical(
                    cat_probs ** (1 / (temperature + 1e-9))
                )
                sampled_latent = cat_distribution.sample()
                current_tokens[~current_mask] = sampled_latent[~current_mask]

                y = step_output["inter output"]
                z = step_output["latent"]

            yield current_tokens

            if step_output:
                if torch.all(step_output["stop"] > 0):
                    print(f"step output: {step_output['stop']}")
                    break

            if (~current_mask).sum() == 0:
                break
