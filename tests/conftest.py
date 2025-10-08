from typing import Any
import pytest

import torch
from torch.utils.data import Dataset, TensorDataset

from loaders.base import Stage
from loaders.datasets import MnistDataset
from loss.aggregators import WeightedSumAggregator
from utils.learning import LearningParameters


@pytest.fixture
def dataset() -> MnistDataset:
    return MnistDataset(stage=Stage.TRAIN, data_path="data", preload=False)


class MockDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"images": torch.rand(1, 28, 28), "class": torch.randint(0, 10, (1,))}

    def __len__(self) -> int:
        return self.length


@pytest.fixture
def mock_dataset_train() -> Dataset:
    return MockDataset(10)


@pytest.fixture
def mock_dataset_val() -> Dataset:
    return MockDataset(5)


@pytest.fixture
def learning_params() -> LearningParameters:
    return LearningParameters(
        model_name="test",
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=2,
        epochs=1,
        beta_ema=0.999,
        gradient_clip=1.0,
        save_path="saved",
        amp=False,
        val_split=0.2,
        test_split=0.1,
        devices=[0],
        num_workers=4,
        loss_monitor="validation total loss",
        interval="step",
        frequency=1,
        trigger_loss=0.0,
    )


class DummyLossComponent:
    def __init__(self, name: str, weight: float, differentiable: bool):
        self.name = name
        self.weight = weight
        self.differentiable = differentiable

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.tensor(1.0)


@pytest.fixture
def loss_aggregator() -> WeightedSumAggregator:
    component1 = DummyLossComponent("component1", 0.5, True)
    component2 = DummyLossComponent("component2", 0.3, False)
    component3 = DummyLossComponent("component3", 0.2, True)
    components = [component1, component2, component3]
    return WeightedSumAggregator(components)
