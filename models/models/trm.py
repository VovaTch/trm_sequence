import math
import torch
import torch.nn as nn

from .backbone.base import Core


class TinyRecursiveModel(nn.Module):
    """
    General TRM implementation, with support for latent recursion, deep recursion, and a general forward function
    """

    def __init__(
        self,
        core: Core,
        z_loop: int,
        y_loop: int,
        input_embedding: nn.Module,
        output_head: nn.Module,
        q_head: nn.Module,
    ) -> None:
        """
        Initializer

        Args:
            core (Core): core module
            z_loop (int): number of latent reasoning steps
            y_loop (int): number of deep recursion steps
            input_embedding (nn.Module): input embedding module
            output_head (nn.Module): output head from the network output
            q_head (nn.Module): network cut-off head
        """
        super().__init__()
        self._core = core
        self._z_loop = z_loop
        self._y_loop = y_loop
        self._input_embedding = input_embedding
        self._output_head = output_head
        self._q_head = q_head

    @property
    def core(self) -> Core:
        """
        Returns the core underlying model of the TRM.
        """
        return self._core

    def latent_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Latent recursion method of the TRM.

        Args:
            input (torch.Tensor): Input tensor
            output (torch.Tensor): Output tensor
            latent (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor and latent tensor
        """
        for _ in range(self._z_loop):
            _, latent = self._core(input, output, latent)
        output, _ = self._core(None, output, latent)
        return output, latent

    def deep_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deep recursion method of the TRM.

        Args:
            input (torch.Tensor): Input tensor
            output (torch.Tensor): Output tensor
            latent (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor, latent tensor, output head tensor, and stop tensor
        """
        input = self._input_embedding(input)
        with torch.no_grad():
            for _ in range(self._y_loop - 1):
                output, latent = self.latent_recursion(input, output, latent)
        output, latent = self.latent_recursion(input, output, latent)
        return (
            output.detach(),
            latent.detach(),
            self._output_head(output),
            self._q_head(output),
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method of the TRM.

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Output tensor
            z (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor and latent tensor
        """
        _, _, output, latent = self.deep_recursion(x, y, z)
        return output, latent

    @torch.inference_mode()
    def verbose_latent_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        A variation of the latent recursion method that also outputs all the intermediate latents

        Args:
            input (torch.Tensor): Input tensor
            output (torch.Tensor): Output tensor
            latent (torch.Tensor): Latent tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]: returns the outputs, last latents, and all previous latents
        """
        all_latents = []
        for _ in range(self._z_loop):
            _, latent = self._core(input, output, latent)
            all_latents.append(latent)

        output, _ = self._core(None, output, latent)
        return output, latent, all_latents

    @torch.inference_mode()
    def verbose_deep_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        float,
    ]:
        """
        Deep recursion method of the TRM, also outputs all output and latent history

        Args:
            input (torch.Tensor): Input tensor
            output (torch.Tensor): Output tensor
            latent (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor, latent tensor, output head tensor, stop tensor
                output history and latent history
        """
        input = self._input_embedding(input)
        total_latents = []
        total_outputs = []
        with torch.no_grad():
            for _ in range(self._y_loop - 1):
                output, latent, all_latents = self.verbose_latent_recursion(
                    input, output, latent
                )
                total_latents.extend(all_latents)
                total_outputs.append(output)
        output, latent, all_latents = self.verbose_latent_recursion(
            input, output, latent
        )
        total_latents.extend(all_latents)
        total_outputs.append(output)

        head_outputs = self._output_head(output)
        output_probs = head_outputs[:, -1, :].softmax(dim=-1)
        entropy = -torch.sum(output_probs * torch.log(output_probs + 1e-10), dim=-1)
        certainty = 1 - (entropy / math.log(output_probs.shape[-1]))

        return (
            output.detach(),
            latent.detach(),
            self._output_head(output),
            self._q_head(output),
            total_latents,
            total_outputs,
            certainty[0].item(),  # FIXME: temporary fix for single batch certainty
        )

    @torch.inference_mode()
    def verbose_forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor], float
    ]:
        """
        Forward method of the TRM, also outputs all output and latent history

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Output tensor
            z (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor, latent tensor, output head tensor, stop tensor
                latent history and output history
        """
        _, _, output, latent, all_latents, all_outputs, certainty = (
            self.verbose_deep_recursion(x, y, z)
        )
        return output, latent, all_outputs, all_latents, certainty
