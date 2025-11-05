"""Trainer modules for training, model management and logging."""

import argparse

import torch.nn as nn

from .train import Trainer


def init_trainer(model: nn.Module, args: argparse.Namespace) -> Trainer:
    """Initialize the Trainer with the given model and arguments.

    Args:
        model (nn.Module): The language model to be trained.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Trainer: An instance of the Trainer class.
    """
    trainer_kwargs = vars(args).copy()
    trainer_kwargs.pop("model", None)
    return Trainer(model=model, **trainer_kwargs)


__all__ = ["Trainer"]
