import torch
import torch.nn as nn


class LinearNeuronModel(nn.Module):
    def __init__(self, history_depth: int) -> None:
        super().__init__()
        self._linear = nn.Linear(history_depth, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._linear(x)
        return out
