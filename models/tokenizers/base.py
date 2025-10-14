from abc import ABC, abstractmethod
from typing import Sequence


class CustomTokenizer(ABC):
    @abstractmethod
    def create_mappings(self, text: str) -> tuple[dict[int, str], dict[str, int]]: ...

    @abstractmethod
    def has_mapping(self) -> bool: ...

    @abstractmethod
    def set_mapping(
        self, text2token: dict[str, int], token2text: dict[int, str]
    ) -> None: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, token_idx: Sequence[int]) -> str: ...
