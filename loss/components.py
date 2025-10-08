from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


class LossComponent(ABC):
    """
    Basic loss component base class, using the __call__ method to compute the loss.

    Attributes:
        name (str): The name of the loss component.
        differentiable (bool): Whether the loss component is differentiable.
        weight (float): The weight of the loss component.
    """

    name: str
    differentiable: bool
    weight: float

    @abstractmethod
    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the loss.

        Args:
            pred (dict[str, torch.Tensor]): A dictionary containing the predicted values.
            target (dict[str, torch.Tensor]): A dictionary containing the target values.

        Returns:
            torch.Tensor: The computed loss as a tensor.
        """
        ...


@dataclass
class BasicClassificationLoss(LossComponent):
    """
    Basic classification loss, most commonly cross entropy.

    Args:
        name (str): The name of the loss.
        weight (float): The weight of the loss.
        base_loss (nn.Module): The base loss function.
        differentiable (bool, optional): Whether the loss is differentiable. Defaults to True.

    Returns:
        torch.Tensor: The computed loss value.
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        return self.base_loss(pred["logits"], target["class"])


@dataclass
class ReconstructionLoss(LossComponent):
    """
    Reconstruction loss of slices or images most commonly.

    Args:
    *   name (str): The name of the loss.
    *   weight (float): The weight of the loss.
    *   base_loss (nn.Module): The base loss function.
    *   rec_key (str): The key for accessing the reconstruction values in the prediction and target dictionaries.
    *   transform_func (Callable[[torch.Tensor], torch.Tensor], optional):
        The transformation function to apply to the reconstruction values. Defaults to lambda x: x.
    *   differentiable (bool, optional): Whether the loss is differentiable. Defaults to True.

    Returns:
        torch.Tensor: The computed loss value.
    """

    name: str
    weight: float
    base_loss: nn.Module
    rec_key: str
    transform_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        return self.base_loss(
            self.transform_func(pred[self.rec_key]),
            self.transform_func(target[self.rec_key]),
        )


@dataclass
class PercentCorrect(LossComponent):
    """
    Percent correct metric for classification tasks.

    Args:
        name (str): The name of the metric.
        weight (float): The weight of the metric.
        differentiable (bool, optional): Whether the metric is differentiable. Defaults to False.

    Returns:
        torch.Tensor: The computed metric value.
    """

    name: str
    weight: float
    differentiable: bool = False

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the metric.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The computed metric value.
        """
        pred_logits_argmax = torch.argmax(pred["logits"], dim=1)
        correct = torch.sum(pred_logits_argmax == target["class"])
        return correct / torch.numel(pred_logits_argmax)
