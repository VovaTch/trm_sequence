from typing import Any, Protocol, Sequence

import torch
import torch.nn as nn

from loss.aggregators import LossOutput
from utils.containers import LearningParameters
from .base import BaseLightningModule, LossAggregator


class ITokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_idx: Sequence[int]) -> str: ...


class AutoRegressorModule(BaseLightningModule):
    """
    Module designed to handle Mamba/Transformer model training.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_params: LearningParameters,
        tokenizer: ITokenizer | None = None,
        val_interval: int = 16,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
        eval_text: str = "The meaning of life is ",
        certainty_threshold: float = 0.8,
    ) -> None:
        super().__init__(
            model,
            learning_params,
            transforms,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )
        self._val_count = -1
        self._val_interval = val_interval
        self._tokenizer = tokenizer
        self._eval_text = eval_text
        self._certainty_threshold = certainty_threshold
        self._log_weights = False

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the autoregressor module.

        Args:
            input (dict[str, torch.Tensor]): Input dictionary containing the indices.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing the model predictions.
        """
        tokens = input["tokens"]
        log_step = self.global_step if self.training and hasattr(self, '_log_weights') and self._log_weights else None
        outputs = self.model(tokens, tokens.shape[1], log_step=log_step)
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
        self._log_weights = (phase == "train" and self.global_step % 100 == 0)
        
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return None
        loss = self.loss_aggregator(output, batch)
        loss_total = self.handle_loss(loss, phase)
        
        if self._log_weights and hasattr(self.model, 'set_tensorboard_writer'):
            try:
                tensorboard = self.logger.experiment
                self.model.set_tensorboard_writer(tensorboard)
            except Exception:
                pass
        
        return loss_total

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
        
        if hasattr(self.model, 'set_tensorboard_writer'):
            self.model.set_tensorboard_writer(tensorboard)
        
        self._log_eval_post_activations(batched_tokenized_text[:1], tensorboard)

    @torch.inference_mode()
    def generate_next_tokens(
        self, input_seq: torch.Tensor, temperature: float = 0.7, top_k: int = 0
    ) -> torch.Tensor:
        """
        Generates a single token batch from input sequences with the same length.

        Args:
            input_seq (torch.Tensor): The input sequence.
            temperature (float, optional): The temperature of the softmax distribution. Defaults to 0.7.
            top_k (int, optional): The number of top tokens to consider. Defaults to 0.
        """
        if temperature < 0:
            raise ValueError(f"Temperature must be non negative, got {temperature}")
        outputs = self.model(input_seq, 1, certainty_stop=True)  # Expected BS x 1 x C
        outputs = outputs[-1]

        if top_k > 0:
            values, _ = torch.topk(outputs, min(top_k, outputs.shape[-1]), dim=2)
            kth_value = values[:, -1].unsqueeze(-1)
            outputs = torch.where(outputs < kth_value, -float("inf"), outputs)

        pre_softmax = outputs / (temperature + 1e-8)
        probs = torch.softmax(pre_softmax, dim=2)
        dist = torch.distributions.Categorical(probs)
        sampled_tokens = dist.sample()
        return sampled_tokens

    @torch.inference_mode()
    def generate(
        self,
        input_seq: torch.Tensor,
        max_seq_length: int,
        temperature: float = 0.7,
        top_k: int = 0,
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
        for _ in range(max_seq_length - len(input_seq[-1])):
            next_tokens = self.generate_next_tokens(output_seq, temperature, top_k)
            output_seq = torch.cat(
                (output_seq, next_tokens),
                dim=1,
            )
        return output_seq
    
    def _log_eval_post_activations(self, input_seq: torch.Tensor, tensorboard) -> None:
        """
        Log post-activations during evaluation.
        
        Args:
            input_seq: Input sequence tensor (BS x L)
            tensorboard: Tensorboard writer
        """
        try:
            with torch.no_grad():
                result = self.model(input_seq, 1, certainty_stop=True, return_post_activations=True)
                if isinstance(result, tuple):
                    outputs, post_activations = result
                    
                    for tick_idx, post_act in enumerate(post_activations):
                        prefix = f"eval/tick_{tick_idx}"
                        
                        tensorboard.add_histogram(f"{prefix}/post_activations", post_act, self._val_count)
                        tensorboard.add_scalar(f"{prefix}/post_activations_mean", post_act.mean(), self._val_count)
                        tensorboard.add_scalar(f"{prefix}/post_activations_std", post_act.std(), self._val_count)
                        tensorboard.add_scalar(f"{prefix}/post_activations_max", post_act.max(), self._val_count)
                        tensorboard.add_scalar(f"{prefix}/post_activations_min", post_act.min(), self._val_count)
                        
                        per_neuron_mean = post_act.mean(dim=(0, 1))
                        per_neuron_std = post_act.std(dim=(0, 1))
                        tensorboard.add_histogram(f"{prefix}/per_neuron_mean", per_neuron_mean, self._val_count)
                        tensorboard.add_histogram(f"{prefix}/per_neuron_std", per_neuron_std, self._val_count)
                    
                    all_post_acts = torch.stack(post_activations, dim=0)
                    tensorboard.add_histogram("eval/all_ticks/post_activations", all_post_acts, self._val_count)
                    tensorboard.add_scalar("eval/num_ticks", len(post_activations), self._val_count)
                    
        except Exception as e:
            pass
