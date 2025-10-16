import torch
import torch.nn as nn
from functools import partial

from models.models.backbone.rope import RotaryEmbedding
from models.models.backbone.sin_pos_enc import (
    SinusoidalPositionEmbeddings,
    apply_pos_encoding,
)


class TokenDiffusionTransformer(nn.Module):
    """
    Simple diffusion LLM implementation without fancy optimizations, using Pytorch's implementation.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout

        self._token_embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        # self._pos_embedding = RotaryEmbedding(hidden_dim)
        pos_embedding_obj = SinusoidalPositionEmbeddings(hidden_dim)
        self._pos_embedding = partial(
            apply_pos_encoding, pos_encoding=pos_embedding_obj
        )

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

        self._mlp_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        # self._mlp_head.weight = nn.Parameter(self._token_embedding.weight[:-1, ...])

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len = input.shape

        if mask is None:
            mask = torch.zeros((batch_size, seq_len), device=input.device)
        mask = mask.clone()

        masked_inputs = input.clone()
        masked_inputs[~mask] = self._vocab_size

        token_embedding = self._token_embedding.forward(masked_inputs)
        token_embedding = self._pos_embedding(token_embedding)

        t_outputs = self._transformer_encoder(token_embedding)
        return self._mlp_head(t_outputs)
