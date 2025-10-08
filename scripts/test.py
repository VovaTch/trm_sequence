import os
import hydra
from omegaconf import DictConfig
import torch

from models.base import BaseLightningModule, load_inner_model_state_dict
from utils.learning import get_trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Set seed and precision
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")

    # Get weights path
    weights_path = os.path.join(cfg.learning.save_path, cfg.model_name + ".ckpt")

    # Get loader
    data_module = hydra.utils.instantiate(cfg.data)

    # Get lightning module
    module: BaseLightningModule = hydra.utils.instantiate(
        cfg.module, _convert_="partial"
    ).to("cuda")
    if cfg.use_torch_compile:
        module.model = torch.compile(module.model)  # type: ignore
    module = load_inner_model_state_dict(module, weights_path)

    # Get trainer
    learning_params = hydra.utils.instantiate(cfg.learning)
    trainer = get_trainer(learning_params)  # type: ignore

    # Fit model
    trainer.test(module, data_module)


if __name__ == "__main__":
    main()
