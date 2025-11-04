import argparse
import os
import time
from typing import Generator

from loaders.base import Stage
from loaders.datasets.fineweb import FinewebKarpathyDataset
from models.tokenizers.rustbpe import RustBPETokenizer
from utils.logger import LOGGER


def _text_iterator(
    dataset: FinewebKarpathyDataset, max_chars: int, text_bs: int
) -> Generator[str, None, None]:

    num_chars = 0
    for batch in dataset._parquet_iter_batched():
        for doc in batch:
            doc_text = doc
            if len(doc_text) > text_bs:
                doc_text = doc_text[:text_bs]
            num_chars += len(doc_text)
            yield doc_text
            if num_chars > max_chars:
                return


def main(args: argparse.Namespace) -> None:

    os.makedirs(args.output_path, exist_ok=True)

    dataset = FinewebKarpathyDataset(
        args.text_batch_size,
        1024,
        Stage.TRAIN,
        num_files=args.max_chars // args.text_batch_size,
    )
    LOGGER.info("Starting tokenizer training...")
    time_s = time.time()
    iterator = _text_iterator(dataset, args.max_chars, args.text_batch_size)
    tokenizer = RustBPETokenizer.train_from_iterator(iterator, args.vocab_size)
    LOGGER.info(
        f"Finished training the tokenizer in {time.time() - time_s:.3f} seconds"
    )
    tokenizer.save(args.output_path)
    LOGGER.info(f"Saved mappings into {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--max_chars",
        type=int,
        default=1e9,
        help="Maximum characters to train on",
    )
    parser.add_argument(
        "-b", "--text_batch_size", type=int, default=1e5, help="Document batch size"
    )
    parser.add_argument(
        "-v", "--vocab_size", type=int, default=65536, help="Vocabulary size"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=os.path.join("data", "fineweb_karpathy"),
    )
    args = parser.parse_args()
    main(args)
