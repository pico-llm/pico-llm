"""Utility functions for pico-llm."""

from .generation import generate
from .parser import parse_args

__all__ = ["parse_args", "generate"]
