import logging
import os
import time
from multiprocessing import Pool

from pyarrow import parquet as pq
from torch.utils.data import Dataset
import requests
from multiprocessing import Pool

from models.tokenizers.base import CustomTokenizer
from utils.logger import LOGGER

MAX_SHARD = 1822


class FinewebKarpathyDataset(Dataset):
    def __init__(
        self,
        data: str = os.path.join("data", "fineweb_karpathy"),
        base_url: str = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main",
        max_shard: int = 1822,
        tokenizer: CustomTokenizer | None = None,
        download_retries: int = 5,
        chunk_size: int = 1024 * 1024,
        num_files: int = -1,
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

        self._tokenizer = tokenizer  # TODO: temporary before adding Karpathy's tokenizer to the mix, it should be similar to GPT4's

        os.makedirs(data, exist_ok=True)

        self._index_to_filename = lambda index: f"shard_{index:05d}.parquet"

        ids_to_download = list(
            range(MAX_SHARD + 1 if num_files == -1 else min(num_files, MAX_SHARD + 1))
        )

        logger.info(f"Downloading {len(ids_to_download)} files")

        with Pool(processes=num_workers) as pool:
            results = pool.map(self._download_single_file, ids_to_download)

        successful = sum(1 for success in results if success)
        logger.info(f"Successfully downloaded {successful} files")

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
