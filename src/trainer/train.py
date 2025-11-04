"""Training module for language models."""

import torch
import torch.nn as nn

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from trainer.base import BaseTrainer


class Trainer(BaseTrainer):
    """Trainer class for language model training."""
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        optimizer_class: str = "adamw",
        scheduler_class: str = "cosine",
        **kwargs: dict,
    ) -> "Trainer":
        """Initialize the Trainer.

        Args:
            model (nn.Module): The language model to be trained.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_class (str): Optimizer type.
            scheduler_class (str): Learning rate scheduler type.
            **kwargs (dict): Additional arguments for the base trainer.

        Returns:
            Trainer: An instance of the Trainer class.
        """
        super().__init__(
            model, learning_rate, optimizer_class, scheduler_class, **kwargs
        )

    def get_model_attrs(self, model: nn.Module) -> dict:
        """Get model attributes for logging.

        Args:
            model (nn.Module): The language model.

        Returns:
            dict: A dictionary of model attributes.
        """
        attrs = model.__dict__.copy()
        attrs.update({"model_type": type(model).__name__})
        attrs.update({"model_class": model.__class__.__name__})
        attrs.update({"model_module": model.__class__.__module__})
        attrs.update({"total_params": sum(p.numel() for p in model.parameters())})
        return attrs
    
    def train(self, model: nn.Module, train_dataloader: DataLoader, epochs):
        pass


