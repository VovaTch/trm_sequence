from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Core(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, x: torch.Tensor | None, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...
