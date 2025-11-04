"""Utility functions for parsing command-line arguments."""

import argparse


def parse_args()-> argparse.Namespace:
    """Parse command-line arguments for training k-gram or sequence-based models.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files."
    )
    parser.add_argument(
        "--input-files",
        nargs="*",
        default=None,
        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).",
    )
    parser.add_argument(
        "--tinystories-weight",
        type=float,
        default=0.5,
        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).",
    )
    parser.add_argument(
        "--encoding-name",
        type=str,
        default="gpt2",
        help="Name of the tiktoken encoding to use. Default='gpt2'.",
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "kgram_mlp", "transformer"],
        type=str,
        default="lstm",
        help="Model architecture to use. Default='lstm'.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training. Default=16."
    )
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="Number of training epochs. Default=5."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate for optimizer. Default=3e-4."
    )
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=20_000,
        help="Number of training sequences to use (for quick tests). Default=20000.",
    )
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=100,
        help="Log training loss every N steps. Default=100.",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=int,
        default=30,
        help="Generate sample text every N seconds during training. Default=30.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="If set, each epoch ends after this many steps (for quick tests).",
    )
    parser.add_argument(
        "--num-inner-mlp-layers",
        type=int,
        default=1,
        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.",
    )
    parser.add_argument(
        "--monosemantic-enabled",
        action="store_true",
        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.",
    )
    parser.add_argument(
        "--kgram-k",
        type=int,
        default=3,
        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.",
    )
    parser.add_argument(
        "--kgram-chunk-size", type=int, default=1, help="Process k-gram timesteps in micro-batches. Default=1."
    )
    parser.add_argument(
        "--block-size", type=int, default=1024, help="Maximum sequence length for each example. Default=1024."
    )
    parser.add_argument(
        "--embed-size",
        type=int,
        default=1024,
        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=1024,
        help="Dimension of the hidden layers for LSTM, MLP, etc. Default=1024.",
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a", help="Prompt used for generation. Default='Once upon a'."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility. Default=42."
    )

    return parser.parse_args()
