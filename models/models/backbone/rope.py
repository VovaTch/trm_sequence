import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()

        if dim % 2:
            raise ValueError(f"Dimension must be divisible by 2, got {dim}")

        inv_freq = 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).type_as(inv_freq)
        frequencies = torch.einsum("i,j->ij", t, inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)

        # Get cached sin and cos values
        self.register_buffer("cached_cos", embeddings.cos(), persistent=False)
        self.register_buffer("cached_sin", embeddings.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape  # Assuming BxSxD dims

        cos = self.cached_cos[:seq_len, :]  # type: ignore
        sin = self.cached_sin[:seq_len, :]  # type: ignore

        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        return torch.cat((x1, x2), dim=-1)
