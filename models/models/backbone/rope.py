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

    def forward(self, x: torch.Tensor, offset: int | None = None) -> torch.Tensor:
        """
        Forward method for RoPE embeddings; offset is unused
        """

        seq_len = x.shape[1]  # Assuming BxSxD dims

        if x.dim() == 3:
            cos = self.cached_cos[:seq_len, :]  # type: ignore
            sin = self.cached_sin[:seq_len, :]  # type: ignore
        elif x.dim() == 4:
            # Handle batch_size x num_heads x seq_len x dim case
            cos = self.cached_cos[None, :seq_len, None, :]  # type: ignore
            sin = self.cached_sin[None, :seq_len, None, :]  # type: ignore

        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        return torch.cat((x1, x2), dim=-1)
