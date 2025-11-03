import os
import pickle
from typing import Generator, Self

import tiktoken

import rustbpe

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>",  # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",  # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RustBPETokenizer:
    def __init__(self, encoder, bos_token: str) -> None:
        self._encoder = encoder
        self._bos_token = bos_token

    @classmethod
    def train_from_iterator(
        cls, text_iterator: Generator[str, None, None], vocab_size: int
    ) -> Self:
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()  # type: ignore
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert (
            vocab_size_no_special >= 256
        ), f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
        )
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens,  # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> Self:
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name: str) -> Self:
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")
