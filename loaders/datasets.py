from __future__ import annotations
from typing import Any

from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms.functional as TF

from .base import Stage


class MnistDataset(Dataset):
    """
    Basic Mnist dataset, loads either the training or the validation sets (for val and test.)
    """

    def __init__(self, stage: Stage, data_path: str, preload: bool = False) -> None:
        """
        Constructor method

        Args:
            stage (Stage): Stage of the training
            data_path (str): Path of the data for the model
            preload (bool, optional): Pre-load the model, here it's unused. Defaults to False.
        """
        super().__init__()
        if stage == Stage.TRAIN:
            self.base_dataset = datasets.MNIST(data_path, train=True, download=True)
        elif stage == Stage.VALIDATION or stage == Stage.TEST:
            self.base_dataset = datasets.MNIST(data_path, train=False, download=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Standard Pytorch getitem method, gets a dictionary of tensors as a data.

        Args:
            index (int): Index of data to get

        Returns:
            dict[str, Any]: A data dictionary with 2 entries:
                -   'images' for image data
                -   'class' for ground truth classes
        """
        data = self.base_dataset.__getitem__(index)
        return {"images": TF.to_tensor(data[0]), "class": data[1]}

    def __len__(self) -> int:
        """
        Length of the dataset

        Returns:
            int: Dataset length
        """
        return len(self.base_dataset)
