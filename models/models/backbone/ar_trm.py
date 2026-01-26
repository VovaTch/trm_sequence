# TODO: KV cache

from enum import Enum
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models.backbone.base import Core
from models.models.experimental.snake import Snake
from utils.other import rms_norm

from .rope import RotaryEmbedding


class EmbeddingType(Enum):
    ROPE = "rope"
    SIN = "sin"


def norm(x: torch.Tensor) -> torch.Tensor:
    """
    Pure RMS norm without learned parameters.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Normalized tensor
    """
    return F.rms_norm(x, (x.size(-1),))


class FullSelfAttention(nn.Module):
    """
    Full attention implementation from Andrej Karpathy's Nanochat
    """

    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        n_embd: int,
        layer_idx: int,
        max_seq_len: int = 2048,
        is_causal: bool = True,
    ) -> None:
        """
        Initializes the attention module

        Args:
            n_head (int): Number of attention heads
            n_kv_head (int): Number of key/value heads
            n_embd (int): Embedding dimension
            layer_idx (int): Layer index
            max_seq_len (int, optional): Maximum sequence length. Defaults to 2048.
            is_causal (bool, optional): Whether to use causal attention. Defaults to True.
        """
        super().__init__()

        self.layer_idx = layer_idx
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        self._max_seq_len = max_seq_len
        self._is_causal = is_causal

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self._rot_embedding_q = RotaryEmbedding(self.head_dim, max_seq_len)
        self._rot_embedding_k = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        q, k = self._rot_embedding_q(q), self._rot_embedding_k(k)  # QK rotary embedding
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=self._is_causal)
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    Multi-layer perceptron implementation from Karpathy's Nanochat
    """

    def __init__(self, n_embd: int, mlp_multiplier: int = 4) -> None:
        """
        Initializes the MLP class

        Args:
            n_embd (int): Embedding dimension
            mlp_multiplier (int, optional): Multiplier for the number of units in the MLP. Defaults to 4.
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, mlp_multiplier * n_embd, bias=False)
        self.c_proj = nn.Linear(mlp_multiplier * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class SnakeMLP(nn.Module):
    def __init__(self, n_embd: int, mlp_multiplier: int = 4) -> None:
        """
        Initializes the MLP class

        Args:
            n_embd (int): Embedding dimension
            mlp_multiplier (int, optional): Multiplier for the number of units in the MLP. Defaults to 4.
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, mlp_multiplier * n_embd, bias=False)
        self.c_proj = nn.Linear(mlp_multiplier * n_embd, n_embd, bias=False)
        self._snake = Snake(mlp_multiplier * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    Transformer block implementation from Karpathy's Nanochat
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: int,
        layer_idx: int,
        max_seq_len: int = 2048,
        mlp_multiplier: int = 4,
        is_causal: bool = True,
        activation_type: Literal["normal", "snake"] = "normal",
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the Transformer Block class
        """
        super().__init__()
        self.attn = FullSelfAttention(
            num_heads, num_kv_heads, hidden_dim, layer_idx, max_seq_len, is_causal
        )
        match activation_type:
            case "normal":
                self.mlp = MLP(hidden_dim, mlp_multiplier)
            case "snake":
                self.mlp = SnakeMLP(hidden_dim, mlp_multiplier)
            case _:
                raise ValueError(
                    f"Unknown activation type: {activation_type}, supported types: normal, snake"
                )
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(norm(self._dropout(x)))
        x = x + self.mlp(norm(self._dropout(x)))
        return x


class ARTransformerTRM(Core):
    """
    Diffusion Transformer TRM core for using it for the TRM model
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        seq_delimiter: int = 4096,
        dropout: float = 0.1,
        vocab_size: int = 65,
        mlp_type: Literal["normal", "snake"] = "normal",
    ) -> None:
        """
        Initializer

        Args:
            hidden_dim (int): The hidden dimension of the model.
            vocab_size (int): The vocabulary size of the model.
            latent_len (int): The latent length of the model.
            num_layers (int): The number of layers in the model.
            num_heads (int): The number of heads in the model.
            seq_delimiter (int): The sequence delimiter for the model, defaults to 4096.
                Used for positional embedding of the output and the latent. Sequence length
                should not exceed it.
            dropout (float): The dropout rate for the model, defaults to 0.1.
        """
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._seq_delimiter = seq_delimiter
        self._vocab_size = vocab_size
        self._mlp_type = mlp_type

        self._transformer_encoder = nn.Sequential(
            *[
                Block(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    layer_idx=i,
                    is_causal=True,
                    dropout=dropout,
                    activation_type=mlp_type,
                )
                for i in range(num_layers)
            ]
        )

        self._y_init = nn.Buffer(torch.randn((1, 1, hidden_dim)), persistent=True)
        self._z_init = nn.Buffer(torch.randn((1, 1, hidden_dim)), persistent=True)

    @property
    def y_init(self) -> nn.Buffer:
        return self._y_init

    @property
    def z_init(self) -> nn.Buffer:
        return self._z_init

    def forward(
        self, x: torch.Tensor | None, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if x is not None:

            input_sum = x + y + z

        else:

            input_sum = y + z

        transformer_output = self._transformer_encoder(input_sum)
        output = rms_norm(transformer_output, 1e-6)

        return output, output

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
