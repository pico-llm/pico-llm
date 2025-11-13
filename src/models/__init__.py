"""Implementations of various language models."""

import argparse
import inspect

import torch
import torch.nn as nn

from .kgram_mlp import KGramMLPSeqModel
from .lstm import LSTMSeqModel

MODEL_REGISTRY = {
    "lstm": LSTMSeqModel,
    "kgram_mlp": KGramMLPSeqModel,
}


def _get_model_class(model_name: str) -> type:
    """Retrieve the model class based on the model name.

    Args:
        model_name (str): Name of the model architecture.

    Returns:
        class: Corresponding model class.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    raise ValueError(f"Model '{model_name}' is not recognized. Available models: {list(MODEL_REGISTRY.keys())}")


def _construct_init_args(cls: type, args: argparse.Namespace) -> dict:
    """Construct a dictionary of initialization arguments for the given class based on the provided argparse.Namespace.

    Args:
        cls (type): The class for which to construct initialization arguments.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: A dictionary of initialization arguments.
    """
    sig = inspect.signature(cls.__init__)
    param_names = set(sig.parameters.keys())
    return {k: v for k, v in vars(args).items() if k in param_names and k != "self"}


def init_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> nn.Module:
    """Initialize the model based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        vocab_size (int): Size of the vocabulary.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: Instantiated model.
    """
    model_cls = _get_model_class(args.model)
    print(f"Using model: {args.model}, cls: {model_cls}")
    if args.checkpoint is not None:
        return model_cls.from_pretrained(args.checkpoint).to(device)
    init_args = _construct_init_args(model_cls, args)
    init_args.setdefault("vocab_size", vocab_size)
    return model_cls(**init_args).to(device)


__all__ = ["init_model"]
