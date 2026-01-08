import math
from typing import Any, TypedDict
import torch
import torch.nn as nn


from loss.aggregators import LossAggregator, LossOutput
from models.models.trm import TinyRecursiveModel
from models.modules.autoregressor import ITokenizer
from models.modules.base import BaseLightningModule
from utils.containers import LearningParameters


class TokenGenerationOutput(TypedDict):
    output: torch.Tensor
    inter_output: torch.Tensor
    latent: torch.Tensor


class VerboseGenerationOutput(TypedDict):
    tokens: torch.Tensor
    all_latents: list[list[torch.Tensor]]
    all_outputs: list[list[torch.Tensor]]


class ARLanguageTRMModule(BaseLightningModule):
    """
    Lightning module for the MNIST classifier, extends the BaseLightningModule class.
    """

    def __init__(
        self,
        model: TinyRecursiveModel,
        learning_params: LearningParameters,
        supervision_steps: int = 4,
        gradient_clip: float = 0.1,
        tokenizer: ITokenizer | None = None,
        val_interval: int = 16,
        training_certainty_cutoff: float = 0.9,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        eval_text: str = "The meaning of life is ",
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
        self._core_hidden_dim = model.core.hidden_dim
        self._gradient_clip = gradient_clip
        self._val_interval = val_interval
        self._tokenizer = tokenizer
        self._val_count = -1
        self._eval_text = eval_text
        self._certainty_cutoff = training_certainty_cutoff

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

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        y_init = self.model.core.y_init.repeat(
            (batch["tokens"].shape[0], batch["tokens"].shape[1], 1)
        ).to(batch["tokens"].device)
        z_init = self.model.core.z_init.repeat(
            (batch["tokens"].shape[0], batch["tokens"].shape[1], 1)
        ).to(batch["tokens"].device)

        y = y_init
        z = z_init

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        total_loss_output = LossOutput(torch.zeros(1).to(batch["tokens"].device), {})

        for _ in range(self._supervision_steps):
            sup_step_output = self.forward(
                {"input": batch["tokens"], "inter output": y, "latent": z}
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

            if self._certainty_cutoff <= 0.0:
                continue
            probs = torch.softmax(sup_step_output["logits"], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_entropy = math.log(sup_step_output["logits"].shape[-1])
            certainty = 1 - (entropy / max_entropy)
            if torch.all(certainty >= self._certainty_cutoff):
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
            log_name = f"{phase}/{name.replace('_', ' ')}"
            self.log(
                log_name,
                loss.individual[name],
                batch_size=self.learning_params.batch_size,
                sync_dist=True,
            )
        self.log(
            f"{phase}/total loss",
            loss.total,
            prog_bar=True,
            batch_size=self.learning_params.batch_size,
            sync_dist=True,
        )
        return loss.total

    @torch.inference_mode()
    def generate_next_tokens(
        self,
        input_seq: torch.Tensor,
        temperature: float = 0.7,
        top_k: int = 0,
        y: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        certainty_cutoff: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a single token batch from input sequences with the same length.

        Args:
            input_seq (torch.Tensor): The input sequence.
            temperature (float, optional): The temperature of the softmax distribution. Defaults to 0.7.
            top_k (int, optional): The number of top tokens to consider. Defaults to 0.
        """
        if temperature < 0:
            raise ValueError(f"Temperature must be non negative, got {temperature}")

        if y is None:
            y_init = self.model.core.y_init.repeat(
                (input_seq.shape[0], input_seq.shape[1], 1)
            ).to(input_seq.device)
            y = y_init
        elif input_seq.shape[1] != y.shape[1]:
            y_init = self.model.core.y_init.repeat(
                (input_seq.shape[0], input_seq.shape[1] - y.shape[1], 1)
            )
            y = torch.cat((y, y_init), dim=1)

        if z is None:
            z_init = self.model.core.z_init.repeat(
                (input_seq.shape[0], input_seq.shape[1], 1)
            ).to(input_seq.device)
            z = z_init
        elif input_seq.shape[1] != z.shape[1]:
            z_init = self.model.core.z_init.repeat(
                (input_seq.shape[0], input_seq.shape[1] - z.shape[1], 1)
            )
            z = torch.cat((z, z_init), dim=1)

        outputs = None
        for _ in range(self._supervision_steps):
            sup_step_output = self.forward(
                {"input": input_seq, "inter output": y, "latent": z}
            )
            outputs = sup_step_output["output"]
            y = sup_step_output["inter output"]
            z = sup_step_output["latent"]
            if torch.all(sup_step_output["stop"][:, -1] > 0):
                break

            if certainty_cutoff > 0:
                probs = torch.softmax(outputs[:, -1, :], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                max_entropy = math.log(outputs.shape[-1])
                certainty = 1 - (entropy / max_entropy)
                if torch.all(certainty >= certainty_cutoff):
                    break

        assert outputs is not None

        if top_k > 0:
            values, _ = torch.topk(outputs, min(top_k, outputs.shape[-1]), dim=2)
            kth_value = values[:, :, -1].unsqueeze(-1)
            outputs = torch.where(outputs < kth_value, -float("inf"), outputs)

        pre_softmax = outputs / (temperature + 1e-8)
        probs = torch.softmax(pre_softmax, dim=2)
        dist = torch.distributions.Categorical(probs)
        sampled_tokens = dist.sample()
        return sampled_tokens[:, -1:], y, z

    @torch.inference_mode()
    def generate(
        self,
        input_seq: torch.Tensor,
        max_seq_length: int,
        temperature: float = 0.7,
        top_k: int = 0,
        certainty_cutoff: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate a sequence with length max_seq_length given an input sequence

        Args:
            input_seq (torch.Tensor): The input sequence.
            max_seq_length (int): The maximum length of the generated sequence.
            temperature (float, optional): The temperature of the softmax distribution. Defaults to 0.7.
            top_k (int, optional): The number of top tokens to consider. Defaults to 0.

        Returns:
            torch.Tensor: The generated sequence.
        """
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)
        output_seq = input_seq

        y = None
        z = None

        for _ in range(max_seq_length - len(input_seq[-1])):
            next_tokens, y, z = self.generate_next_tokens(
                output_seq, temperature, top_k, y, z, certainty_cutoff=certainty_cutoff
            )
            output_seq = torch.cat(
                (output_seq, next_tokens),
                dim=1,
            )
        return output_seq

    @torch.inference_mode()
    def verbose_generate_next_tokens(
        self,
        input_seq: torch.Tensor,
        temperature: float = 0.7,
        top_k: int = 0,
        y: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]
    ]:
        """
        Generates a single token batch with verbose output (all latents and outputs).

        Args:
            input_seq (torch.Tensor): The input sequence.
            temperature (float, optional): The temperature of the softmax distribution. Defaults to 0.7.
            top_k (int, optional): The number of top tokens to consider. Defaults to 0.
            y (torch.Tensor | None, optional): The intermediate output tensor. Defaults to None.
            z (torch.Tensor | None, optional): The latent tensor. Defaults to None.

        Returns:
            tuple: sampled_tokens, y, z, all_latents, all_outputs
        """
        if temperature < 0:
            raise ValueError(f"Temperature must be non negative, got {temperature}")

        if y is None:
            y_init = self.model.core.y_init.repeat(
                (input_seq.shape[0], input_seq.shape[1], 1)
            ).to(input_seq.device)
            y = y_init
        elif input_seq.shape[1] != y.shape[1]:
            y_init = self.model.core.y_init.repeat(
                (input_seq.shape[0], input_seq.shape[1] - y.shape[1], 1)
            )
            y = torch.cat((y, y_init), dim=1)

        if z is None:
            z_init = self.model.core.z_init.repeat(
                (input_seq.shape[0], input_seq.shape[1], 1)
            ).to(input_seq.device)
            z = z_init
        elif input_seq.shape[1] != z.shape[1]:
            z_init = self.model.core.z_init.repeat(
                (input_seq.shape[0], input_seq.shape[1] - z.shape[1], 1)
            )
            z = torch.cat((z, z_init), dim=1)

        all_latents: list[torch.Tensor] = []
        all_outputs: list[torch.Tensor] = []
        outputs = None

        for _ in range(self._supervision_steps):
            y, z, y_hat, q_hat, step_latents, step_outputs = (
                self.model.verbose_deep_recursion(input_seq, y, z)
            )
            all_latents.extend(step_latents)
            all_outputs.extend(step_outputs)
            outputs = y_hat
            if torch.all(q_hat[:, -1] > 0):
                break

        assert outputs is not None

        if top_k > 0:
            values, _ = torch.topk(outputs, min(top_k, outputs.shape[-1]), dim=2)
            kth_value = values[:, :, -1].unsqueeze(-1)
            outputs = torch.where(outputs < kth_value, -float("inf"), outputs)

        pre_softmax = outputs / (temperature + 1e-8)
        probs = torch.softmax(pre_softmax, dim=2)
        dist = torch.distributions.Categorical(probs)
        sampled_tokens = dist.sample()
        return sampled_tokens[:, -1:], y, z, all_latents, all_outputs

    @torch.inference_mode()
    def verbose_generate(
        self,
        input_seq: torch.Tensor,
        max_seq_length: int,
        temperature: float = 0.7,
        top_k: int = 0,
    ) -> VerboseGenerationOutput:
        """
        Generate a sequence with verbose output containing all latents and outputs.

        Args:
            input_seq (torch.Tensor): The input sequence.
            max_seq_length (int): The maximum length of the generated sequence.
            temperature (float, optional): The temperature of the softmax distribution. Defaults to 0.7.
            top_k (int, optional): The number of top tokens to consider. Defaults to 0.

        Returns:
            VerboseGenerationOutput: Contains tokens, all_latents, all_outputs
        """
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)
        output_seq = input_seq

        y = None
        z = None

        all_latents: list[list[torch.Tensor]] = []
        all_outputs: list[list[torch.Tensor]] = []

        for _ in range(max_seq_length - len(input_seq[-1])):
            next_tokens, y, z, step_latents, step_outputs = (
                self.verbose_generate_next_tokens(output_seq, temperature, top_k, y, z)
            )
            all_latents.append(step_latents)
            all_outputs.append(step_outputs)
            output_seq = torch.cat(
                (output_seq, next_tokens),
                dim=1,
            )

        return VerboseGenerationOutput(
            tokens=output_seq,
            all_latents=all_latents,
            all_outputs=all_outputs,
        )

    def on_validation_start(self) -> None:
        """
        Creates a sequence of text for eval during training purposes
        """
        self._val_count += 1
        if self._val_count % self._val_interval != 0 or self._tokenizer is None:
            return

        tokenized_text = self._tokenizer.encode(self._eval_text)
        batched_tokenized_text = [tokenized_text] * self.learning_params.batch_size
        batched_tokenized_text = torch.tensor(batched_tokenized_text).to(self.device)

        generated_text = self.generate(batched_tokenized_text, 256)
        decoded_texts = [
            self._tokenizer.decode(tokens.tolist()) for tokens in generated_text
        ]

        tensorboard = self.logger.experiment  # type: ignore
        for idx, decoded_text in enumerate(decoded_texts):
            tensorboard.add_text(
                f"Text sample {idx + 1}", decoded_text, global_step=self._val_count
            )
