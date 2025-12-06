import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import rms_norm


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        pos_embedding: nn.Module,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._pos_embedding = pos_embedding

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}"
            )

        self._head_dim = hidden_dim // num_heads
        self._is_causal = is_causal

        self._proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = q.size()  # TODO: figure out why the dim is 3

        # Split to heads
        q = q.view(batch_size, -1, self._num_heads, self._head_dim)
        k = k.view(batch_size, -1, self._num_heads, self._head_dim)
        v = v.view(batch_size, -1, self._num_heads, self._head_dim)

        # Pos embedding
        # q = self._pos_embedding(q)
        k = self._pos_embedding(k)
        q, k = rms_norm(q), rms_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self._is_causal)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        y = self._proj(y)
        return y
