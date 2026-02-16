"""
Manifold-Constrained Hyper-Connections (mHC) variant of the AR Transformer TRM core.

Implements the mHC paper (arXiv:2512.24880, DeepSeek-AI) which replaces standard residual
connections with an n-stream expanded residual using three learnable mappings (H_pre, H_post,
H_res) projected onto constrained manifolds for stable training.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models.backbone.base import Core
from models.models.backbone.ar_trm import Block
from utils.other import rms_norm


def sinkhorn_knopp(h_tilde: torch.Tensor, t_max: int) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for projecting a matrix onto the Birkhoff polytope
    (set of doubly stochastic matrices).

    Args:
        H_tilde: Unconstrained matrix (..., n, n)
        t_max: Number of normalization iterations

    Returns:
        Doubly stochastic matrix (..., n, n)
    """
    m = torch.exp(h_tilde)
    for _ in range(t_max):
        m = m / m.sum(dim=-1, keepdim=True)
        m = m / m.sum(dim=-2, keepdim=True)
    return m


class mHCMapping(nn.Module):
    """
    Computes the three mHC mappings (H_pre, H_post, H_res) for a single layer.

    Each mapping is composed of a dynamic (input-dependent) part and a static bias,
    then projected onto the appropriate manifold:
    - H_pre: sigmoid (non-negative, for aggregating n streams to 1)
    - H_post: 2*sigmoid (non-negative, for expanding 1 stream to n)
    - H_res: Sinkhorn-Knopp doubly stochastic (for stable residual mixing)
    """

    def __init__(
        self,
        n: int,
        c: int,
        t_max: int = 20,
        alpha_init: float = 0.01,
    ) -> None:
        """
        Args:
            n: Expansion rate (number of residual streams)
            C: Per-stream hidden dimension
            t_max: Sinkhorn-Knopp iterations
            alpha_init: Initial value for gating scalars
        """
        super().__init__()
        self._n = n
        self._c = c
        self._t_max = t_max
        nC = n * c

        # Dynamic projection matrices
        self.phi_pre = nn.Parameter(torch.randn(nC, n) * 0.01)
        self.phi_post = nn.Parameter(torch.randn(nC, n) * 0.01)
        self.phi_res = nn.Parameter(torch.randn(nC, n * n) * 0.01)

        # Static biases
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.zeros(n, n))

        # Gating scalars (initialized small per paper)
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))

    def forward(
        self, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mHC mappings from the flattened n-stream state.

        Args:
            x_flat: Flattened input (B, T, nC)

        Returns:
            H_pre: (B, T, n) — aggregation weights
            H_post: (B, T, n) — expansion weights
            H_res: (B, T, n, n) — residual mixing (doubly stochastic)
        """
        # RMSNorm on the flattened input
        x_norm = F.rms_norm(x_flat, (x_flat.size(-1),))

        # Unconstrained mappings: dynamic (input-dependent) + static bias
        h_tilde_pre = self.alpha_pre * (x_norm @ self.phi_pre) + self.b_pre
        h_tilde_post = self.alpha_post * (x_norm @ self.phi_post) + self.b_post
        h_tilde_res = self.alpha_res * (x_norm @ self.phi_res) + self.b_res.flatten()

        # Reshape res to matrix form
        B, T, _ = x_flat.shape
        h_tilde_res = h_tilde_res.view(B, T, self._n, self._n)

        # Manifold projections
        h_pre = torch.sigmoid(h_tilde_pre)
        h_post = 2.0 * torch.sigmoid(h_tilde_post)
        h_res = sinkhorn_knopp(h_tilde_res, self._t_max)

        return h_pre, h_post, h_res


class mHCBlock(nn.Module):
    """
    A single transformer block wrapped with mHC residual connections.

    Replaces the standard residual `x + F(x)` with the mHC formulation:
    `x_{l+1} = H_res @ x_l + H_post^T * F(H_pre * x_l)`

    where the mappings are computed dynamically from the input and projected
    onto constrained manifolds.
    """

    def __init__(
        self,
        n: int,
        c: int,
        num_heads: int,
        num_kv_heads: int,
        layer_idx: int,
        max_seq_len: int = 2048,
        mlp_multiplier: int = 4,
        is_causal: bool = True,
        activation_type: Literal["normal", "snake"] = "normal",
        dropout: float = 0.1,
        sinkhorn_iterations: int = 20,
        alpha_init: float = 0.01,
    ) -> None:
        """
        Args:
            n: Expansion rate (number of residual streams)
            C: Per-stream hidden dimension (the inner Block operates at this width)
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads
            layer_idx: Layer index for the inner attention block
            max_seq_len: Maximum sequence length for RoPE
            mlp_multiplier: MLP hidden dim multiplier
            is_causal: Whether attention is causal
            activation_type: MLP activation type
            dropout: Dropout rate
            sinkhorn_iterations: Number of Sinkhorn-Knopp iterations
            alpha_init: Initial value for gating scalars
        """
        super().__init__()
        self._n = n
        self._c = c

        self._inner_block = Block(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_dim=c,
            layer_idx=layer_idx,
            max_seq_len=max_seq_len,
            mlp_multiplier=mlp_multiplier,
            is_causal=is_causal,
            activation_type=activation_type,
            dropout=dropout,
        )
        self._mapping = mHCMapping(
            n=n, c=c, t_max=sinkhorn_iterations, alpha_init=alpha_init
        )

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one mHC block.

        Args:
            x_l: Input tensor (B, T, n, C)

        Returns:
            Output tensor (B, T, n, C)
        """
        B, T, n, C = x_l.shape

        x_flat = x_l.reshape(B, T, n * C)
        h_pre, h_post, h_res = self._mapping(x_flat)
        h_in = (h_pre.unsqueeze(-1) * x_l).sum(dim=-2)  # (B, T, C)
        h_out = self._inner_block(h_in)  # (B, T, C)
        res = torch.einsum("btij,btjc->btic", h_res, x_l)  # (B, T, n, C)
        expansion = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)  # (B, T, n, C)

        return res + expansion


class mHCARTransformerTRM(Core):
    """
    Autoregressive Transformer TRM core with Manifold-Constrained Hyper-Connections.

    Replaces the standard residual connections in ARTransformerTRM with mHC,
    expanding the residual stream by a factor of n and using constrained learnable
    mappings to maintain training stability.

    The `hidden_dim` parameter represents the full expanded width (n * C), where C
    is the per-stream dimension used by the inner transformer blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        expansion_rate: int = 4,
        sinkhorn_iterations: int = 20,
        alpha_init: float = 0.01,
        seq_delimiter: int = 4096,
        dropout: float = 0.1,
        vocab_size: int = 65,
        mlp_type: Literal["normal", "snake"] = "normal",
    ) -> None:
        """
        Args:
            hidden_dim: Full expanded hidden dimension (n * C). Must be divisible by expansion_rate.
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads (operates at per-stream dim C)
            expansion_rate: Number of residual streams (n)
            sinkhorn_iterations: Number of Sinkhorn-Knopp iterations for H_res projection
            alpha_init: Initial value for gating scalars in mHC mappings
            seq_delimiter: Maximum sequence length for positional embeddings
            dropout: Dropout rate
            vocab_size: Vocabulary size
            mlp_type: MLP activation type ("normal" or "snake")
        """
        super().__init__()

        assert (
            hidden_dim % expansion_rate == 0
        ), f"hidden_dim ({hidden_dim}) must be divisible by expansion_rate ({expansion_rate})"

        self._hidden_dim = hidden_dim
        self._expansion_rate = expansion_rate
        self._c = hidden_dim // expansion_rate
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._seq_delimiter = seq_delimiter
        self._vocab_size = vocab_size

        self._blocks = nn.ModuleList(
            [
                mHCBlock(
                    n=expansion_rate,
                    c=self._c,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    layer_idx=i,
                    max_seq_len=seq_delimiter,
                    is_causal=True,
                    activation_type=mlp_type,
                    dropout=dropout,
                    sinkhorn_iterations=sinkhorn_iterations,
                    alpha_init=alpha_init,
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
        """
        Forward pass for the mHC AR Transformer TRM core.

        Args:
            x: Input tensor (B, T, hidden_dim) or None
            y: Output tensor (B, T, hidden_dim)
            z: Latent tensor (B, T, hidden_dim)

        Returns:
            Tuple of (output, output), both (B, T, hidden_dim)
        """
        if x is not None:
            input_sum = x + y + z
        else:
            input_sum = y + z

        B, T, _ = input_sum.shape

        # Reshape to n-stream form: (B, T, n, C)
        state = input_sum.view(B, T, self._expansion_rate, self._c)

        # Pass through mHC blocks
        for block in self._blocks:
            state = block(state)

        # Flatten back to (B, T, hidden_dim)
        output = state.reshape(B, T, self._hidden_dim)
        output = rms_norm(output, 1e-6)

        return output, output

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
