"""Trainer modules for training, model management and logging."""

import argparse

import torch.nn as nn
from torch.utils.data import DataLoader

from .train import Trainer


def init_trainer(model: nn.Module, train_loader: DataLoader, args: argparse.Namespace) -> Trainer:
    """Initialize the Trainer with the given model and arguments.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The training dataloader.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Trainer: An instance of the Trainer class.
    """
    if args.checkpoint is not None:
        return Trainer.from_pretrained(args.checkpoint, model=model)
    trainer_kwargs = vars(args).copy()
    trainer_kwargs.pop("model", None)  # TODO: I don't remember why I did this lol, check if needed
    trainer_kwargs.setdefault("num_steps", len(train_loader) * args.num_epochs)
    return Trainer(model=model, **trainer_kwargs)


__all__ = ["Trainer"]
