from __future__ import annotations

from enum import Enum
from functools import partial

import torch
import torch.nn as nn

from .base import Core
from .rope import RotaryEmbedding
from .sin_pos_enc import SinusoidalPositionEmbeddings, apply_pos_encoding


class EmbeddingType(Enum):
    ROPE = "rope"
    SIN = "sin"


class DiffusionTransformerTRM(Core):
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
        embedding_type: EmbeddingType = EmbeddingType.SIN,
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
            embedding_type (EmbeddingType): The embedding type for the model.
        """
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._seq_delimiter = seq_delimiter
        self._embedding_type = embedding_type
        self._vocab_size = vocab_size

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            norm_first=True,
        )
        self._transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_layers,
        )

        match embedding_type:
            case EmbeddingType.ROPE:
                self._pos_embedding = RotaryEmbedding(hidden_dim)
            case EmbeddingType.SIN:
                pos_embedding_obj = SinusoidalPositionEmbeddings(hidden_dim)
                self._pos_embedding = partial(
                    apply_pos_encoding, pos_encoding=pos_embedding_obj
                )

    def forward(
        self, x: torch.Tensor | None, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        y = self._pos_embedding(y, offset=self._seq_delimiter)
        z = self._pos_embedding(z, offset=self._seq_delimiter * 2)

        if x is not None:

            x_after_pos = self._pos_embedding(x)

            transformer_input = torch.cat((x_after_pos, y, z), dim=1)
            transformer_output = self._transformer_encoder(transformer_input)
            _, y_out, z_out = torch.split(
                transformer_output, [x.shape[1], y.shape[1], z.shape[1]], dim=1
            )

        else:

            transformer_input = torch.cat((y, z), dim=1)
            transformer_output = self._transformer_encoder(transformer_input)
            y_out, z_out = torch.split(
                transformer_output, [y.shape[1], z.shape[1]], dim=1
            )

        return y_out, z_out

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class InputEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._embedding = nn.Embedding(vocab_size + 1, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding(x)


class LinearOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        self._head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


class LinearQOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, seq_length: int) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._seq_length = seq_length

        layers = []
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Linear(seq_length, 1))

        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layers[0](x)
        x = x.view(x.shape[0], -1)
        return self._layers[1](x)
