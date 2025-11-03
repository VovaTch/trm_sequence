from loaders.base import Stage
from loaders.datasets.fineweb import FinewebKarpathyDataset

MAX_IDX = 5


def main() -> None:
    fineweb_dataset = FinewebKarpathyDataset(4, 1024, Stage.TRAIN, num_files=2)

    for idx, batch in enumerate(fineweb_dataset):
        print(batch["tokens"])
        if (idx + 1) % MAX_IDX == 0:
            break


if __name__ == "__main__":
    main()
