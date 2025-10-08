import hydra
from omegaconf import DictConfig
import torch

from models.base import load_inner_model_state_dict
from utils.learning import get_trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Set precision
    torch.set_float32_matmul_precision("high")

    # Get loader
    data_module = hydra.utils.instantiate(cfg.data)

    # Get lightning module
    module = hydra.utils.instantiate(cfg.module, _convert_="partial").to("cuda")
    if cfg.use_torch_compile:
        module.model = torch.compile(module.model)
    if cfg.resume is not None:
        module = load_inner_model_state_dict(module, cfg.resume)

    # Get trainer
    learning_params = hydra.utils.instantiate(cfg.learning)
    trainer = get_trainer(learning_params)  # type: ignore

    # Fit model
    trainer.fit(module, data_module)


if __name__ == "__main__":
    main()
