import torch

from utils.sample_schedulers.base import SampleScheduler


class MaxThresholdScheduler(SampleScheduler):
    def __init__(self, threshold: float, seq_len: int = 1024) -> None:
        super().__init__()
        self._threshold = threshold
        self._step_num = 0
        self._seq_len = seq_len

    def sample(
        self, step: int, prev_mask: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample a new mask based on the maximum logit values exceeding a threshold.

        Args:
            step (int): The current step in the sampling process.
            prev_mask (torch.Tensor): The previous mask tensor, size BS x L
            logits (torch.Tensor): The logits tensor from the model, size Bs x L x Vocab

        Returns:
            torch.Tensor: The new mask tensor after applying the thresholding logic.
        """
        new_mask = prev_mask.clone()

        probs = torch.softmax(logits, dim=-1)

        probs_maxed, _ = torch.max(probs, dim=-1)  # BS x L, BS x L
        to_unmask = probs_maxed >= self._threshold
        new_mask[to_unmask] = True
        newly_unmasked = to_unmask & (~prev_mask)
        print(
            f"Num of threshold logits: {newly_unmasked.sum().item()}, step num {self._step_num}"
        )

        probs_maxed[prev_mask] = 0
        batch_probs_maxed, batch_logit_indices = torch.max(
            probs_maxed, dim=-1
        )  # BS, BS
        batch_threshold_reached = batch_probs_maxed >= self._threshold
        for b in range(new_mask.shape[0]):
            if not batch_threshold_reached[b]:
                index = batch_logit_indices[b].item()
                new_mask[b, index] = True

        self._step_num += 1

        return new_mask

    @property
    def num_steps(self) -> int:
        return self._seq_len
