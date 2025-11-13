"""Utility functions for parsing command-line arguments."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training k-gram or sequence-based models.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files."
    )

    # Data and dataset parameters
    parser.add_argument(
        "--input-files",
        nargs="*",
        default=None,
        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).",
    )
    # TODO: Figure out why we need this, if we can set a better default, etc.
    parser.add_argument(
        "--tinystories-weight",
        type=float,
        default=0.5,
        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).",
    )
    parser.add_argument(
        "--train-subset-size",
        type=int,
        help="Number of training sequences to use. Default=None (use all data).",
    )
    parser.add_argument(
        "--block-size", type=int, default=1024, help="Maximum sequence length for each example. Default=1024."
    )

    # Tokenizer and model parameters
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
    # Transformer-specific hyperparameters (used when --model transformer)
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Hidden dimension (d_model) for Transformer (default=128).",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads for Transformer (default=4).",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=2,
        help="Number of Transformer blocks (default=2).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length / positional embeddings for Transformer (default=1024).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for Transformer (default=0.1).",
    )
    #pre-norm by default, set this flag to use post-norm
    parser.add_argument(
        "--no-pre-norm",
        action="store_false",
        dest="pre_norm",
        default=True,
        help="If set, disables pre-norm and uses post-norm (original Transformer). By default pre-norm is used.",
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
        "--k",
        type=int,
        default=3,
        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1, help="Process k-gram timesteps in micro-batches. Default=1."
    )
    parser.add_argument(
        "--num-inner-layers",
        type=int,
        default=1,
        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.",
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=["full", "scaled", "onehot"],
        default="full",
        help="Type of input representation for k-gram MLP. Options are 'full', 'scaled', 'onehot'. Default='full'.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path or model ID of a pre-trained checkpoint to load."
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training. Default=16.")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs. Default=5.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for optimizer. Default=3e-4.")
    parser.add_argument(
        "--optimizer-class",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer class to use. Default='adamw'.",
    )
    parser.add_argument(
        "--scheduler-class",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "exponential"],
        help="Learning rate scheduler class to use. Default='cosine'.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Ratio of warmup steps to total training steps for the scheduler. Default=0.1.",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=100,
        help="Log training loss every N steps. Default=100.",
    )
    parser.add_argument(
        "--save-interval-steps",
        type=int,
        default=200,
        help="Save model checkpoint every N steps. Default=200.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./saved_models",
        help="Directory to save model checkpoints. Default='./saved_models'.",
    )
    parser.add_argument(
        "--save-latest",
        action="store_true",
        help="If set, overwrite the latest checkpoint instead of saving per step.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="If set, track and save the best model checkpoint based on training loss.",
    )

    # Generation parameters for training samples
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a",
        help="Prompt used for generation while training. Default='Once upon a'.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate during training samples. Default=50.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability for generation during training. Default=0.9.",
    )

    # Weights & Biases integration
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="If set, use Weights & Biases for experiment tracking.",
    )
    parser.add_argument("--wandb-entity", type=str, default="pico-llm", help="Weights & Biases entity name.")
    parser.add_argument("--wandb-project", type=str, default="training", help="Weights & Biases project name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Weights & Biases run name.")

    # HuggingFace Hub integration
    parser.add_argument(
        "--upload-model-to-hub", action="store_true", help="If set, upload the model to Hugging Face Hub."
    )
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face Hub repository ID.")

    # System parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Default=42.")

    # Optional features
    parser.add_argument(
        "--monosemantic-analysis",
        action="store_true",
        help="If set, run the monosemantic analysis.",
    )

    return parser.parse_args()
