import torch
import torch.nn as nn

from .base import Core


class MLPCore(Core):
    """
    Fully connected core for the TRM model
    """

    def __init__(
        self,
        input_dim: int = 28 * 28,
        output_dim: int = 10,
        latent_dim: int = 16,
        num_layers: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._latent_dim = latent_dim
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim

        layers = []
        for idx in range(num_layers):
            if idx == 0:
                layers.append(
                    nn.Linear(input_dim + output_dim + latent_dim, hidden_dim)
                )
            elif idx == num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.RMSNorm(hidden_dim))
                layers.append(nn.Linear(hidden_dim, latent_dim + output_dim))
            else:
                layers.append(nn.GELU())
                layers.append(nn.RMSNorm(hidden_dim))
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        self._layers = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method; assumes x is at the size of BS x In, y is BS x Out, z is BS x Latent
        """
        if x.shape[0] != y.shape[0] or y.shape[0] != z.shape[0]:
            raise ValueError(
                f"X, Y, and Z must have the same batch size, got x: {x}, y: {y}, z: {z}"
            )

        x = x.flatten(start_dim=1)
        concat_all = torch.cat((x, y, z), dim=1)
        all_out = self._layers(concat_all).to(x.device)
        y_out, z_out = torch.split(all_out, [self._output_dim, self._latent_dim], dim=1)
        return y_out, z_out

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def vocab_size(self) -> int:
        return 0
