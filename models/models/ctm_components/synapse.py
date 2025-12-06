import torch
import torch.nn as nn

from .rms_norm import rms_norm


class SwigLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self._linear1 = nn.Linear(input_dim, hidden_dim * 2, bias=False)
        self._linear2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self._activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self._linear1(x)
        x_a, x_b = x_proj.chunk(2, dim=-1)
        x_activated = self._activation(x_a) * x_b
        x_activated = rms_norm(x_activated)
        out = self._linear2(x_activated)
        return out
