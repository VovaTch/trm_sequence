from __future__ import annotations
from math import floor
import os
from typing import Any, TypedDict
import requests
import logging

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms.functional as TF
import numpy as np

from models.tokenizers.base import CustomTokenizer
from models.tokenizers.char import CharLevelTokenizer

from .base import Stage
from utils.logger import LOGGER


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


class TextSequenceOutput(TypedDict):
    text: str
    tokens: torch.Tensor


class CharLevelTS(Dataset):
    """
    Dataset class for Tiny Shakespeare, character-level tokenization for small language model testing.
    """

    def __init__(
        self,
        path: str = os.path.join("data", "tiny_shakespeare", "raw.txt"),
        tokenizer: CustomTokenizer | None = None,
        tokenized_data_path: str = os.path.join("data", "tiny_shakespeare"),
        stage: Stage = Stage.TRAIN,
        val_split: float = 0.09,
        test_split: float = 0.01,
        sequence_length: int = 1024,
        logger: logging.Logger = LOGGER,
    ) -> None:

        if test_split + val_split > 1.0:
            raise ValueError(
                f"The val and test split must not exceed 1.0 together, got {val_split} for val and {test_split} for test"
            )
        train_split = 1 - test_split - val_split

        self._tokenizer = (
            CharLevelTokenizer(tokenized_data_path) if tokenizer is None else tokenizer
        )

        super().__init__()
        self._path = path
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(path, "wb") as f:
                f.write(requests.get(data_url).content)
        self._logger = logger
        self.sequence_length = sequence_length

        match stage:
            case Stage.TRAIN:
                tokenized_file_path = os.path.join(tokenized_data_path, "train.bin")
            case Stage.VALIDATION:
                tokenized_file_path = os.path.join(tokenized_data_path, "val.bin")
            case Stage.TEST:
                tokenized_file_path = os.path.join(tokenized_data_path, "test.bin")

        if os.path.isfile(tokenized_file_path):
            self._data = np.memmap(tokenized_file_path, dtype=np.uint16, mode="r")
            self._logger.info(f"Loaded tokenized data from {tokenized_file_path}")
            return

        with open(path, "r") as f:
            data = f.read()

        match stage:
            case Stage.TRAIN:
                split = data[: int(len(data) * train_split)]
            case Stage.VALIDATION:
                split = data[
                    int(len(data) * train_split) : int(
                        len(data) * (train_split + val_split)
                    )
                ]
            case Stage.TEST:
                split = data[int(len(data) * (train_split + val_split)) :]

        self._logger.info(
            f"Loaded tiny Shakespeare dataset with {len(split)} characters"
        )

        if not self._tokenizer.has_mapping():
            token2text, text2token = self._tokenizer.create_mappings(data)
            self._tokenizer.set_mapping(text2token, token2text)
        self._data = self._tokenizer.encode(split)
        self._data = np.array(self._data, dtype=np.uint16)
        self._data.tofile(tokenized_file_path)
        self._logger.info(f"Saved dataset into {tokenized_data_path}")

    def __len__(self) -> int:
        return floor(len(self._data) / self.sequence_length)

    def __getitem__(self, index: int) -> TextSequenceOutput:
        raw = self._data[
            index * self.sequence_length : (index + 1) * self.sequence_length
        ]
        raw_writable = raw.copy()
        assert isinstance(raw_writable, np.ndarray)
        raw_writable.flags.writeable = True
        data = torch.from_numpy(raw_writable.astype(np.int64))
        text = self._tokenizer.decode(list(raw_writable))
        return {"text": text, "tokens": data}


class MaskedTextSequenceOutput(TypedDict):
    text: str
    tokens: torch.Tensor
    mask: torch.Tensor


class MaskedCharLevelTS(CharLevelTS):
    """
    A variation fit for training dataset to LLaDa type of diffusion language model
    """

    def __getitem__(self, index: int) -> MaskedTextSequenceOutput:
        chat_ts_out = super().__getitem__(index)
        prob_to_mask = torch.rand((1,))
        p_mask = torch.ones(chat_ts_out["tokens"].shape[0]) * prob_to_mask
        mask = torch.bernoulli(p_mask).bool()
        return {**chat_ts_out, "mask": mask}
