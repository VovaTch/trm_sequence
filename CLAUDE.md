# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**trm_sequence** is a PyTorch Lightning + Hydra research framework for training autoregressive language models using the **Tiny Recursive Model (TRM)** architecture. The TRM features two novel recursion mechanisms: **latent recursion** (z_loop iterations in latent space) and **deep recursion** (y_loop iterations with gradient detachment). The project includes a Rust-based BPE tokenizer via PyO3/Maturin.

## Common Commands

### Training

```bash
python -m scripts.train --config-name=trm_ar          # AR-TRM on default data
python -m scripts.train --config-name=trm_ar_fineweb   # AR-TRM on FineWeb
python -m scripts.train --config-name=dllm              # Diffusion LLM variant
python -m scripts.train --config-name=lctm_ts           # LCTM time-series
```

Hydra overrides work as usual:

```bash
python -m scripts.train --config-name=trm_ar model.z_loop=8 learning.batch_size=64
python -m scripts.train --multirun --config-name=lctm_ts model.max_thought_step=8,16
```

### Testing

```bash
pytest tests/                        # Run all tests
pytest tests/test_models/            # Run model tests only
pytest tests/test_loss/              # Run loss tests only
pytest tests/path/to/test_file.py -k "test_name"  # Run a single test
```

### Other Scripts

```bash
python -m scripts.test                      # Evaluate on test set
python -m scripts.train_tokenizer           # Train BPE tokenizer from FineWeb data
python -m scripts.visualize_generation      # Generate latent visualization videos
```

### Setup & Build

```bash
uv sync                              # Install all dependencies
maturin develop --release            # Build the Rust BPE tokenizer
```

## Architecture

### Core Recursion (models/models/trm.py)

`TinyRecursiveModel` is the central architecture. It takes a `Core` (the underlying transformer), and applies:

1. **Latent recursion** (`z_loop`): Runs the core repeatedly on the latent state without producing output, then generates output from the final latent.
2. **Deep recursion** (`y_loop`): Wraps latent recursion in an outer loop. Only the last y_loop iteration has gradients; earlier ones use `torch.no_grad()` with detached outputs.

The model has three heads: `input_embedding`, `output_head` (token logits), and `q_head` (stop/cutoff signal).

### Module Hierarchy (models/modules/)

```
BaseLightningModule          # Shared optimizer/scheduler setup via Hydra instantiation
├── ARLanguageTRMModule      # Main AR training loop with deep recursion supervision
├── DiffusionLLMModule       # Diffusion-based language model
├── TRMDiffusionModule       # TRM with diffusion
└── MnistModule              # MNIST classifier (for testing)
```

`ARLanguageTRMModule` (models/modules/trm_ar.py) is the primary module. Its `step()` method calls `deep_recursion` in a loop, computing loss at each step with detached intermediate states.

### Configuration (config/)

Hydra configs use `_target_` for instantiation. Top-level configs (e.g. `trm_ar.yaml`) compose from:

- `data/` — DataModule configs
- `model/` — Model architecture configs
- `module/` — Lightning module configs
- `loss/` — Loss aggregator and component configs
- `optimizer/`, `scheduler/`, `learning/` — Training hyperparameters

### Loss System (loss/)

`WeightedSumAggregator` combines multiple `LossComponent` instances, each with a name, weight, and `differentiable` flag. Non-differentiable components are logged but excluded from backprop.

### Data Loading (loaders/)

`SeparatedSetModule` is the Lightning DataModule. Dataset implementations: `FinewebKarpathyDataset` (language modeling), `MnistDataset`, `TimeSeriesDataset`.

### Tokenizers (models/tokenizers/)

`ITokenizer` protocol defines the interface. Implementations: `CharTokenizer` (character-level), `RustBPETokenizer` (wraps the Rust `rustbpe` module built via Maturin).

### Trainer Factory (utils/learning.py)

`get_trainer()` builds a Lightning Trainer with callbacks: ModelCheckpoint, EarlyStopping, LearningRateMonitor, EMA, ModelSummary. Configured via `LearningParameters` dataclass.

## Key Conventions

- Python 3.12, package managed with `uv`
- All model/data/loss instantiation goes through `hydra.utils.instantiate(cfg.xxx)`
- Training precision: `torch.set_float32_matmul_precision("high")` is always set
- Gitignored directories: `data/`, `saved/`, `outputs/`, `trained/`, `weights/`
- Scripts are run as modules: `python -m scripts.train`, not `python scripts/train.py`

## Coding guidelines

- Strict type hinting; use protocols when applicable. Specify dictionary types, use TypedDict when necessary
- Use of abstract classes, type hinting to refer them
- Dependency injection and inversion when possible
- Descriptive variable names
- Docstrings for the classes
- Refrain from using hard-coded values, make them initialized through Hydra
