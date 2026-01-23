from torch import nn
import torch
from models.models.backbone.ar_trm import ARTransformerTRM, FullSelfAttention, norm
from models.models.experimental.ripple_linear import RippleLinear


class RippleFullAttention(FullSelfAttention):
    """
    Experimental RippleNet version of the attention mechanism with RoPE positional embeddings.
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
        super().__init__(n_head, n_kv_head, n_embd, layer_idx, max_seq_len, is_causal)
        self.c_q = RippleLinear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = RippleLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = RippleLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = RippleLinear(self.n_embd, self.n_embd, bias=False)


class RippleFeedForward(nn.Module):
    """
    Experimental Feed Forward module
    """

    def __init__(self, n_embd: int, mlp_multiplier: int = 2) -> None:
        super().__init__()
        self._c_fc = RippleLinear(n_embd, mlp_multiplier * n_embd)
        self._activation = nn.LeakyReLU(negative_slope=0.1)
        self._c_proj = RippleLinear(mlp_multiplier * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._c_fc(x)
        x = self._activation(x)
        x = self._c_proj(x)
        return x


class RippleBlock(nn.Module):
    """
    Transformer block implementation from Karpathy's Nanochat, spiced up with experimental Ripple blocks
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
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the Transformer Block class
        """
        super().__init__()
        self.attn = RippleFullAttention(
            num_heads, num_kv_heads, hidden_dim, layer_idx, max_seq_len, is_causal
        )
        self.mlp = RippleFeedForward(hidden_dim, mlp_multiplier)
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(norm(self._dropout(x)))
        x = x + self.mlp(norm(self._dropout(x)))
        return x


class ARRippleTRM(ARTransformerTRM):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        seq_delimiter: int = 4096,
        dropout: float = 0.1,
        vocab_size: int = 65,
    ) -> None:
        super().__init__(
            hidden_dim, num_layers, num_heads, seq_delimiter, dropout, vocab_size
        )
        self._transformer_encoder = nn.Sequential(
            *[
                RippleBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    layer_idx=i,
                    is_causal=True,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
