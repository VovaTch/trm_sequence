import os

from pyarrow import parquet as pq
from torch.utils.data import Dataset

from models.tokenizers.base import CustomTokenizer


class FinewebKarpathyDataset(Dataset):
    def __init__(
        self,
        data: str = os.path.join("data", "fineweb_karpathy"),
        base_url: str = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main",
        max_shard: int = 1822,
        tokenizer: CustomTokenizer | None = None,
    ) -> None:
        super().__init__()
        self._data = data
        self._base_url = base_url
        self._max_shard = max_shard

        self._tokenizer = tokenizer  # TODO: temporary before adding Karpathy's tokenizer to the mix, it should be similar to GPT4's

        os.makedirs(data, exist_ok=True)

        self._index_to_filename = lambda index: f"shard_{index:05d}.parquet"

    def _list_parquet_files(self) -> list[str]:
        parquet_files = sorted(
            [
                f
                for f in os.listdir(self._data)
                if f.endswith(".parquet") and not f.endswith(".tmp")
            ]
        )
        parquet_paths = [os.path.join(self._data, f) for f in parquet_files]
        return parquet_paths
