import logging
import torch
import torch.nn as nn

from utils.logger import LOGGER
from .rope import RotaryEmbedding


class AttnBackbone(nn.Module):
    def __init__(
        self,
        vocab_size_l: int = 256,
        vocab_size_h: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self._vocab_size_l = vocab_size_l
        self._vocab_size_h = vocab_size_h
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout = dropout
        self._logger = logger

        self._input_embeddings = nn.Embedding(
            num_embeddings=vocab_size_l + vocab_size_h, embedding_dim=hidden_dim
        )
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, norm_first=True
        )
        self._transformer = nn.TransformerDecoder(
            decoder_layer=transformer_layer,
            num_layers=num_layers,
        )
        self._mlp_head = nn.Linear(hidden_dim, vocab_size_l)

        self._positional_emb = RotaryEmbedding(dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assuming x has the size of BS x L
        """
        embedded = self._input_embeddings(x)
        embedded = self._positional_emb(embedded)
        output = self._transformer(embedded, embedded)
        output = self._mlp_head(output)
        return output
