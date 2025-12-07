from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossComponent


@dataclass
class LLMClassificationLoss(LossComponent):
    """
    Basic classification loss, most commonly cross entropy.

    Args:
        name (str): The name of the loss.
        weight (float): The weight of the loss.
        base_loss (nn.Module): The base loss function.
        ref_key (str): The key for accessing the reference values in the target dictionary.
        pred_key (str): The key for accessing the predicted values in the prediction dictionary.
        differentiable (bool, optional): Whether the loss is differentiable. Defaults to True.
    """

    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str
    ref_key: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        logits = pred[self.pred_key].transpose(1, 2).contiguous()[..., :-1]
        target_indices = target[self.ref_key][..., 1:].long()
        return self.base_loss(logits, target_indices)


@dataclass
class TokenEntropy(LossComponent):
    """
    Computes the entropy of the token distribution.

    Attributes:
        name (str): The name of the loss component.
        weight (float): The weight of the loss component.
        logit_key (str): The key for accessing the logits in the prediction dictionary.
        differentiable (bool): Whether the loss component is differentiable.
    """

    name: str
    weight: float
    logit_key: str
    differentiable: bool = False

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        probs = F.softmax(pred[self.logit_key], dim=2)
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=2).mean()


@dataclass
class LLMPercentCorrect(LossComponent):
    """
    Computes the percentage of correct predictions.

    Attributes:
        name (str): The name of the loss component.
        weight (float): The weight of the loss component.
        logit_key (str): The key for accessing the logits in the prediction dictionary.
        target_key (str): The key for accessing the target values in the target dictionary.
        differentiable (bool): Whether the loss component is differentiable.
    """

    name: str
    weight: float
    pred_key: str
    ref_key: str
    differentiable: bool = False

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        logits = pred[self.pred_key][:, :-1, ...]
        target_indices = target[self.ref_key][..., 1:]
        if logits.dim() == 4:
            target_indices = torch.repeat_interleave(
                target_indices.unsqueeze(-1), logits.shape[-1], -1
            )
        return torch.mean((torch.argmax(logits, dim=2) == target_indices).float())


@dataclass
class CTMLoss(LossComponent):
    """
    Continuous thought machines paper loss; uses both AR classification and an uncertainty measure.

    Attribtues:
        name (str): The name of the loss component.
        weight (float): The weight of the loss component.
        logit_key (str): The key for accessing the logits in the prediction dictionary.
        target_key (str): The key for accessing the target values in the target dictionary.
        differentiable (bool): Whether the loss component is differentiable.
    """

    name: str
    weight: float
    pred_key: str
    ref_key: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        logits = pred[self.pred_key]
        targets = target[self.ref_key]
        if logits.dim() == 4:
            _, _, classes, time_steps = logits.shape  # BS x L x O x Tf
        elif logits.dim() == 3:
            classes = logits.shape[-1]
            logits = logits.unsqueeze(-1)
            time_steps = 1
        elif logits.dim() == 2:
            classes = logits.shape[-1]
            logits = logits.unsqueeze(0).unsqueeze(-1)
            targets = targets.unsqueeze(0)
            time_steps = 1
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")

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
