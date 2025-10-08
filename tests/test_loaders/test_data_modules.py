import pytest
from torch.utils.data import Dataset

from loaders.data_modules import SeparatedSetModule
from utils.learning import LearningParameters


@pytest.fixture
def data_module(
    mock_dataset_train: Dataset,
    mock_dataset_val: Dataset,
    learning_params: LearningParameters,
) -> SeparatedSetModule:
    return SeparatedSetModule(
        learning_params=learning_params,
        train_dataset=mock_dataset_train,
        val_dataset=mock_dataset_val,
    )


def test_train_dataloader(data_module: SeparatedSetModule) -> None:
    train_loader = data_module.train_dataloader()
    assert len(train_loader) == 5


def test_val_dataloader(data_module: SeparatedSetModule) -> None:
    val_loader = data_module.val_dataloader()
    assert len(val_loader) == 3


def test_test_dataloader(data_module: SeparatedSetModule) -> None:
    test_loader = data_module.test_dataloader()
    assert len(test_loader) == 3
