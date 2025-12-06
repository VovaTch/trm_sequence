from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm_components.attn import SelfAttention
from .ctm_components.rope import RotaryEmbedding


class SyncType(Enum):
    ACTION = auto()
    OUTPUT = auto()


class LanguageContinuousThoughtMachine(nn.Module):
    """
    Language version of the CTM, this one will be built for AR language modeling.
    """

    def __init__(
        self,
        model_width: int,
        model_depth: int,
        input_width: int,
        num_attn_heads: int,
        max_thought_step: int,
        sync_dim_action: int,
        sync_dim_output: int,
        vocab_size: int,
        synapse: nn.Module,
        neuron_level_model: nn.Module,
        output_proj: nn.Module,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self._d_width = model_width
        self._m_depth = model_depth
        self._sync_dim_action = sync_dim_action
        self._sync_dim_output = sync_dim_output
        self._max_thought_step = max_thought_step
        self._dropout = dropout
        self._vocab_size = vocab_size

        self._pos_emb = RotaryEmbedding(input_width)
        self._attn = SelfAttention(
            input_width, num_attn_heads, self._pos_emb, is_causal=True
        )
        self._input_embedding = nn.Embedding(vocab_size, input_width)

        self._synapse = synapse
        self._neuron_level_model = neuron_level_model
        self._output_proj = output_proj

        self._z_init = nn.Parameter(torch.randn(model_width))
        self._pre_activation_history_init = nn.Parameter(
            torch.randn((model_width, model_depth))
        )

        self._q_projector = nn.Linear(sync_dim_action, input_width, bias=False)
        self._kv_projector = nn.Linear(input_width, input_width, bias=False)

        self.register_buffer(
            "_idx_left_action",
            torch.randint(0, model_width, size=(self._sync_dim_action,)),
        )
        self.register_buffer(
            "_idx_right_action",
            torch.randint(0, model_width, size=(self._sync_dim_action,)),
        )

        self._sync_exp_decay_action = nn.Parameter(torch.zeros(1, 1, sync_dim_action))

        self.register_buffer(
            "_idx_left_output",
            torch.randint(0, model_width, size=(self._sync_dim_output,)),
        )
        self.register_buffer(
            "_idx_right_output",
            torch.randint(0, model_width, size=(self._sync_dim_output,)),
        )

        self._sync_exp_decay_output = nn.Parameter(
            torch.zeros(1, 1, 1, sync_dim_output)
        )

    def forward(self, x: torch.Tensor, num_output_q: int = 1) -> list[torch.Tensor]:
        """
        Size of x: BS x L
        """
        emb_input = self._input_embedding(x)  # BS x Q x C
        batch_size = x.shape[0]
        kv = self._kv_projector(emb_input)  # BS x Q x C

        pre_activation_history = (
            self._pre_activation_history_init.unsqueeze(0)
            .unsqueeze(0)
            .repeat((batch_size, num_output_q, 1, 1))
            .to(x.device)
        )  # BS x Q x Z x T
        post_activation_history = [
            self._z_init.unsqueeze(0)
            .unsqueeze(0)
            .repeat((batch_size, num_output_q, 1))
            .to(x.device)
        ]  # PAH x BS x Q x Z
        output_history = []

        sync_a = self._compute_sync(
            post_activation_history, SyncType.ACTION
        )  # BS x Q x SA

        z = post_activation_history[0]  # BS x Q x Z

        for _ in range(self._max_thought_step):

            q = self._q_projector(sync_a)  # BS x Q x C
            if q.dim() == 2:
                q = q.unsqueeze(1)
            attn_out = self._attn(q, kv, kv)  # BS x Q x C

            pre_activations = self._synapse(
                torch.cat((attn_out, z), dim=-1)
            )  # BS x Q x (C + Z) => BS x Q x Z
            pre_activation_history = torch.cat(
                (pre_activation_history[..., 1:], pre_activations.unsqueeze(-1)),
                dim=-1,
            )  # BS x Q x Z x T

            z = self._neuron_level_model(pre_activation_history).squeeze(
                -1
            )  # Bs x Q x Z
            post_activation_history.append(z)  # PAH x BS x Q x Z

            sync_a = self._compute_sync(
                post_activation_history, SyncType.ACTION
            )  # BS x Q x SA
            sync_o = self._compute_sync(
                post_activation_history, SyncType.OUTPUT
            )  # BS x Q x SO

            output = self._output_proj(sync_o)  # BS x Q x O
            output_history.append(output)  # PAH x BS x Q x O

        return output_history

    def _compute_sync(
        self, post_activation_history: list[torch.Tensor], sync_type: SyncType
    ) -> torch.Tensor:
        """
        Computes synchronization

        Args:
            post_activation_history (list[torch.Tensor]): Size PAH x BS x Q x Z
            sync_type (SyncType): Sync type, can be action or output
        """

        if len(post_activation_history) == 0:
            raise ValueError("Post activation history is empty.")

        device = post_activation_history[0].device
        num_q = post_activation_history[0].shape[1]

        match sync_type:
            case SyncType.ACTION:
                idx_left = self._idx_left_action
                idx_right = self._idx_right_action
                sync_dim = self._sync_dim_action
                sync_exp_decay = self._sync_exp_decay_action
            case SyncType.OUTPUT:
                idx_left = self._idx_left_output
                idx_right = self._idx_right_output
                sync_dim = self._sync_dim_output
                sync_exp_decay = self._sync_exp_decay_output

        time_steps = len(post_activation_history)

        seq = torch.stack(post_activation_history, dim=1).to(device)  # BS x PAH x Q x Z
        t_back = (
            torch.arange(time_steps - 1, -1, -1).reshape(1, time_steps, 1, 1).to(device)
        )
        exp_decay = torch.exp(-t_back * sync_exp_decay).expand(
            1, time_steps, num_q, sync_dim
        )  # BS x PAH x Q x Z

        seq_multiplied = seq[..., idx_left] * exp_decay * seq[..., idx_right]  # type: ignore
        sync_rep = seq_multiplied.sum(dim=1) / torch.sqrt(exp_decay.sum(dim=1))
        return sync_rep

    def ctm_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _, _, classes, time_steps = logits.shape  # BS x L x O x Tf

        prob = F.softmax(logits, dim=2)
        log_probs = torch.log_softmax(logits, dim=2)
        entropy = -torch.sum(prob * log_probs, dim=2)
        max_entropy = torch.log(torch.tensor(float(classes)))
        certainties = 1 - (entropy / max_entropy)

        targets_expanded = torch.repeat_interleave(
            targets.unsqueeze(-1), time_steps, -1
        )  # TODO: check if correct
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(logits.transpose(1, 2), targets_expanded)

        lowest_idx = losses.argmin(dim=-1)
        certain_idx = certainties.argmax(dim=-1)

        loss = torch.gather(losses, dim=-1, index=lowest_idx.unsqueeze(-1)) / 2
        loss += torch.gather(losses, dim=-1, index=certain_idx.unsqueeze(-1)) / 2

        return loss.mean()
