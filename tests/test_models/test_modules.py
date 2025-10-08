import pytest
import torch
import torch.nn as nn

from loss.aggregators import LossOutput, WeightedSumAggregator
from loss.components import BasicClassificationLoss
from models.models import FCN
from models.modules import MnistClassifierModule
from utils.learning import LearningParameters


@pytest.fixture
def cls_loss_aggregator() -> WeightedSumAggregator:
    cls_loss_component = BasicClassificationLoss(
        "cls_loss", 1.0, torch.nn.CrossEntropyLoss()
    )
    return WeightedSumAggregator([cls_loss_component])


@pytest.fixture
def mnist_classifier_module(
    cls_loss_aggregator: WeightedSumAggregator,
) -> MnistClassifierModule:
    # Create a sample model, learning parameters, and other required objects
    model = FCN()
    learning_params = LearningParameters("testing")
    transforms = nn.Sequential()
    optimizer_cfg = None
    scheduler_cfg = None

    # Create an instance of the MnistClassifierModule
    module = MnistClassifierModule(
        model=model,
        learning_params=learning_params,
        transforms=transforms,
        loss_aggregator=cls_loss_aggregator,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
    )
    return module


def test_forward(mnist_classifier_module: MnistClassifierModule) -> None:
    x = {"images": torch.randn(10, 28, 28)}
    output = mnist_classifier_module.forward(x)
    assert output["logits"].shape == (10, 10)


def test_step(mnist_classifier_module: MnistClassifierModule) -> None:
    batch = {"images": torch.randn(10, 28, 28), "class": torch.randint(0, 10, (10,))}
    phase = "train"
    loss = mnist_classifier_module.step(batch, phase)
    assert loss is None or isinstance(loss, torch.Tensor)


def test_log_loss(mnist_classifier_module: MnistClassifierModule) -> None:
    loss = LossOutput(
        torch.tensor(0.25),
        {"component1": torch.tensor(0.1), "component2": torch.tensor(0.15)},
    )
    phase = "train"
    total_loss = mnist_classifier_module.log_loss(loss, phase)
    assert isinstance(total_loss, torch.Tensor)
