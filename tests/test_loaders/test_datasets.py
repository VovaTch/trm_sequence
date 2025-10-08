from loaders.datasets import MnistDataset


def test_get_item(dataset: MnistDataset) -> None:
    assert dataset[0]["images"].shape == (1, 28, 28)
    assert dataset[0]["class"] in range(10)


def test_len(dataset: MnistDataset) -> None:
    assert len(dataset) == len(dataset.base_dataset)
