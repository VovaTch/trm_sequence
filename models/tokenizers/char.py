import logging
import os
from typing import Sequence
import pickle

from .base import CustomTokenizer
from utils.logger import LOGGER


class CharLevelTokenizer(CustomTokenizer):
    """
    Character level tokenizer for a text body
    """

    def __init__(
        self,
        mapping_path: str = os.path.join("data", "tiny_shakespeare"),
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()

        self._mapping_path = mapping_path
        self._text2token_map = None
        self._token2text_map = None
        self._logger = logger

        os.makedirs(self._mapping_path, exist_ok=True)
        self._meta_file_path = os.path.join(mapping_path, "meta.pkl")

        if not os.path.isfile(self._meta_file_path):
            return

        with open(self._meta_file_path, "rb") as f:
            data = pickle.load(f)

        self._text2token_map = data["text2token_map"]
        self._token2text_map = data["token2text_map"]
        self._logger.info(
            f"Loaded mappings from {self._meta_file_path}, vocabulary size {data['vocab_size']}"
        )

    def has_mapping(self) -> bool:
        return self._text2token_map is not None and self._token2text_map is not None

    def set_mapping(
        self, text2token: dict[str, int], token2text: dict[int, str]
    ) -> None:
        self._text2token_map = text2token
        self._token2text_map = token2text

    @property
    def vocab_size(self) -> int:
        if self._text2token_map is None:
            raise RuntimeError("Mapping not found; create or load a mapping first")
        return len(self._text2token_map)

    def create_mappings(self, text: str) -> tuple[dict[int, str], dict[str, int]]:
        characters = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(characters)}
        itos = {i: ch for i, ch in enumerate(characters)}

        meta = {
            "vocab_size": len(characters),
            "text2token_map": stoi,
            "token2text_map": itos,
        }

        self._logger.info(f"Available vocabulary size: {len(characters)}")

        with open(self._meta_file_path, "wb") as f:
            pickle.dump(meta, f)

        self._logger.info(f"Saved mappings into {self._meta_file_path}")
        return itos, stoi

    def encode(self, text: str) -> list[int]:
        if self._text2token_map is None:
            raise RuntimeError("Mapping not found; create or load a mapping first")
        return [self._text2token_map[ch] for ch in text]

    def decode(self, token_idx: Sequence[int]) -> str:
        if self._token2text_map is None:
            raise RuntimeError("Mapping not found; create or load a mapping first")
        return "".join([self._token2text_map[idx] for idx in token_idx])
