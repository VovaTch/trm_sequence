import logging
import os
import time
from multiprocessing import Pool
from typing import Generator

from pyarrow import parquet as pq
import torch
from torch.utils.data import IterableDataset
import requests

from models.tokenizers.base import CustomTokenizer
from models.tokenizers.rustbpe import RustBPETokenizer
from utils.ddp import get_dist_info
from utils.logger import LOGGER

from ..base import Stage

MAX_SHARD = 1822


class FinewebKarpathyDataset(IterableDataset):
    def __init__(
        self,
        batch_size: int,
        seq_length: int,
        stage: Stage,
        data: str = os.path.join("data", "fineweb_karpathy"),
        base_url: str = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main",
        max_shard: int = 1822,
        tokenizer: CustomTokenizer | None = None,
        download_retries: int = 5,
        chunk_size: int = 1024 * 1024,
        num_files: int = 240,
        num_workers: int = 4,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()
        self._data = data
        self._base_url = base_url
        self._max_shard = max_shard
        self._logger = logger
        self._retries = download_retries
        self._chunk_size = chunk_size
        self._num_files = num_files
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._seq_length = seq_length
        self._stage = stage

        self._tokenizer = (
            tokenizer
            if tokenizer is not None
            else RustBPETokenizer.from_directory(data)
        )

        os.makedirs(data, exist_ok=True)

        ids_to_download = list(
            range(MAX_SHARD + 1 if num_files == -1 else min(num_files, MAX_SHARD + 1))
        )

        logger.info(f"Downloading {len(ids_to_download)} files")

        with Pool(processes=num_workers) as pool:
            results = pool.map(self._download_single_file, ids_to_download)

        successful = sum(1 for success in results if success)
        logger.info(f"Successfully downloaded {successful} files")

        _, self._ddp_rank, _, self._ddp_world_size = get_dist_info()
        logger.info(f"[Rank {self._ddp_rank}] Dataset initialized ")

    def _index_to_filename(self, index: int) -> str:
        return f"shard_{index:05d}.parquet"

    def _list_parquet_files(self) -> list[str]:
        """
        Returns the paths of all .parquet files in the data folder.

        Returns:
            list[str]: All parquet files in the folder
        """
        parquet_files = sorted(
            [
                f
                for f in os.listdir(self._data)
                if f.endswith(".parquet") and not f.endswith(".tmp")
            ]
        )
        parquet_paths = [os.path.join(self._data, f) for f in parquet_files]
        return parquet_paths

    def _download_single_file(self, index: int) -> bool:
        """
        Downloads a single file from huggingface given an index

        Args:
            index (int): The index of the file to download

        Returns:
            bool: True if the file was successfully downloaded, False otherwise
        """
        filename = self._index_to_filename(index)
        filepath = os.path.join(self._data, filename)
        if os.path.exists(filepath):
            self._logger.info(f"File {filepath} already exists. Skipping...")
            return True

        url = f"{self._base_url}/{filename}"
        self._logger.info(f"Downloading {url} to {filepath}")

        for attempt in range(self._retries):
            try:
                response = requests.get(url)
                response.raise_for_status()
                temp_path = filepath + ".tmp"

                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(self._chunk_size):
                        if chunk:
                            f.write(chunk)
                os.rename(temp_path, filepath)
                self._logger.info(f"Downloaded {filename} to {filepath}")
                return True

            except (requests.RequestException, IOError) as e:
                self._logger.error(
                    f"Attempt {(attempt + 1) / self._retries} failed for filename {filename}: {e}"
                )

                # Cleanup
                for path in [filepath + ".tmp", filepath]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as e_in:
                            self._logger.warning(f"Failed to remove: {e_in}")

                if attempt < self._retries:
                    wait_time = 2**attempt
                    self._logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self._logger.error(
                        f"Failed to download {filename} after {self._retries} attempts"
                    )
                    return False

        return False

    def _parquet_iter_batched(
        self, start: int = 0, step: int = 1
    ) -> Generator[list[str], None, None]:
        parquet_paths = self._list_parquet_files()
        parquet_paths = (
            parquet_paths[:-1] if self._stage == Stage.TRAIN else parquet_paths[-1:]
        )

        for file_path in parquet_paths:
            parquet_file = pq.ParquetFile(file_path)
            for row_group_idx in range(start, parquet_file.num_row_groups, step):
                row_group = parquet_file.read_row_group(row_group_idx)
                texts = row_group["text"].to_pylist()
                yield texts

    def _document_batched(self) -> Generator[str, None, None]:
        _, ddp_rank, _, ddp_world_size = get_dist_info()
        while True:
            for batch in self._parquet_iter_batched(
                start=ddp_rank, step=ddp_world_size
            ):
                for line in batch:
                    yield line

    def __iter__(self):

        batch_generator = self._document_batched()
        batch_idx = 0

        assert self._tokenizer is not None

        # main iter loop
        while True:
            token_sample = next(batch_generator)
            tokens_torch = torch.tensor(token_sample, dtype=torch.long)

            prob_to_mask = torch.rand((1,))
            p_mask = torch.ones(tokens_torch.shape[0]) * prob_to_mask
            mask = torch.bernoulli(p_mask).bool()
            batch_idx += 1
            yield {"tokens": tokens_torch, "mask": mask}
