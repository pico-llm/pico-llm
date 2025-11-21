# picoLLM

An educational repository for training and sampling from tiny language models as part of the [CSCI-GA 2565 Machine Learning](https://cs.nyu.edu/courses/fall25/CSCI-GA.2565-001/), Fall 2025 course at NYU. This repository contains dataset utilities, model implementations, training & sampling workflows, and scripts for saving/loading checkpoints and collecting generated outputs.

## Table of contents

- [Project Overview](#project-overview)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
    - [Training](#training)
    - [Sampling and Evaluation](#sampling-and-evaluation)
- [Data](#data)
- [Saved Models & Results](#saved-models--results)
- [Acknowledgements](#acknowledgements)

## Project Overview

Pico-LLM is a minimal experimental codebase for exploring small language models (transformer, LSTM, k-gram MLP variations). The project provides:

- Dataset utilities for converting line-based text into token blocks.
- Model implementations and a simple training loop with logging and checkpointing.
- Sampling utilities to generate text from saved checkpoints and to collect samples across multiple models.
- Integration points for Weights & Biases and the Hugging Face Hub for logging and model hosting.

This repository is intended for teaching and experimentation rather than production use.

## Repository Layout

Key files and folders:

- `src/` — main source code (training, dataset, models, utils, sampling scripts).
	- `pico_llm.py` — high-level training entrypoint script.
	- `sample.py` — script to sample sentences from Hugging Face-hosted checkpoints.
	- `training/` — trainer classes & training utilities.
	- `models/`, `dataset/`, `utils/` — model implementations and helpers.
- `data/` — small example datasets (e.g. `3seqs.txt`).
- `saved_models/` — model checkpoints saved during training (organized by step folders).
- `results/` — sampled outputs and evaluation artifacts (e.g. `sampled_sentences.json`).
- `pyproject.toml` — project metadata and Python packaging info.

## Installation

Use `uv` for environment creation and dependency synchronization (recommended).

1. Install `uv` (if not already installed) from [this](https://docs.astral.sh/uv/getting-started/installation/) link.

2. Create a new project environment with Python 3.12:

```bash
uv venv --python=3.12
```

3. Install dependencies (including all extras defined in the project):

```bash
uv sync --all-extras
```

## Quickstart

### Training

Train a model using the provided training entrypoint. From the repository root you can run:

```bash
python src/pico_llm.py \
	--model transformer \
	--num-epochs 3 \
	--batch-size 16 \
	--save-dir saved_models/step_test \
	--device cuda
```

Important arguments you may want to change:

- `--model`: model type (example: `transformer`, `lstm`, `kgram_mlp`).
- `--num-epochs`: number of epochs to train.
- `--batch-size`: batch size for training.
- `--save-dir`: directory where checkpoints are saved (under `saved_models/`).
- `--device`: `cuda` or `cpu`.
- `--use-wandb`, `--wandb_project`, `--wandb_entity`: enable Weights & Biases logging.
- `--upload_model_to_hub`, `--repo_id`: push checkpoints to the Hugging Face Hub (requires `HF_TOKEN` env var).

Training will periodically save checkpoints into `saved_models/` (e.g. `step_1000`, `best_model`, `final_model`).

### Running on an HPC / SLURM

A ready-to-use SLURM script is provided at `src/slurm/pico_llm.slurm`. This script sets common job parameters and exposes configuration variables near the top of the file so you can customize a run without editing the training code.

How to use it:

- **Edit the top variables**: open `src/slurm/pico_llm.slurm` and set your NYU NetID, working directory (`--chdir`), email, and desired model/training options (batch size, epochs, save directory, etc.).
- **Ensure an environment exists on the cluster**: the script expects a `.venv` activation (`source .venv/bin/activate`) and uses `uv run` to launch training. On the cluster create the environment and install dependencies using `uv` or `pip` as described above. Example (run once on the cluster):

```bash
# create the venv and sync deps (on the cluster)
uv venv --python=3.12
uv sync --all-extras
```

- **Submit the job**: from the repository root on the cluster run:

```bash
sbatch src/slurm/pico_llm.slurm
```

- **Logs and outputs**: the script writes `stdout`/`stderr` to the `logs/` directory (see `#SBATCH --output` / `--error` at the top of the script) and saves checkpoints to the `SAVE_DIR` you configured.

Notes and tips:

- The script uses `uv run src/pico_llm.py` which requires `uv` to be available in the activated environment on the compute node. If your cluster does not allow creating transient environments, consider creating `.venv` with `python -m venv .venv` and installing dependencies with `pip`.
- Make small test runs first (set `NUM_EPOCHS=1`, small `DATASET_SUBSET_SIZE`) to validate configuration before launching long jobs.
- You can run the slurm file as a shell script for debugging (e.g. `bash src/slurm/pico_llm.slurm`).


### Sampling and Evaluation

The `src/sample.py` script collects samples from models (it queries Hugging Face for models authored by `pico-llm` by default). Example usage:

```bash
# Sample 10 sequences using local CUDA (or use --device cpu)
python src/sample.py --device cuda --number 10 --output results/sampled_sentences.json
```

If you want to sample from a local checkpoint rather than the Hugging Face Hub, you can modify the script or use the `models` utilities to load a local `saved_models/` checkpoint.

Notes:
- `sample.py` accepts arguments like `--block-size`, `--top-p` (nucleus sampling values), `--max-new-tokens`, and `--encoding-name`.
- `HF_TOKEN` environment variable should be set if you want to query/upload to Hugging Face Hub.

## Data

- Example data lives in `data/` (for example `data/3seqs.txt`), where each line is treated as an example.
- The dataset utilities in `src/dataset` create token blocks (with a configurable `block_size`) and dataloaders for train/val/test splits.

## Saved Models & Results

- Checkpoints: `saved_models/step_<N>/` contain model weights and training metadata.
- Sampled outputs: `results/` includes JSON outputs such as `sampled_sentences.json` generated by `src/sample.py`.
- To reproduce experiments, keep a copy of the `training_config.json` and `config.json` files that are saved alongside each checkpoint.

## Acknowledgements

This codebase was developed for CSCI-GA.2565 and is intended for educational use and experimentation.
