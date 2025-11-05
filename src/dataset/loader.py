"""Data loading utilities for pico-llm."""

import tiktoken
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from dataset.mixed_sequence_dataset import MixedSequenceDataset


def _seq_collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """Collate function to pad variable-length sequences in a batch.

    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)

    Args:
        batch (list of torch.Tensor): List of 1D LongTensors of various lengths
    Returns:
        torch.Tensor: Padded tensor of shape (max_len, batch_size).
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)
    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq
    return padded


def _load_tinystories(
    tinystories_weight: float, train_subset_size: int, block_size: int, enc: tiktoken.Encoding
) -> list[list[int]]:
    """Load TinyStories dataset from HuggingFace and tokenize sequences.

    Args:
        tinystories_weight (float): Probability of sampling from TinyStories.
        train_subset_size (int): Number of training sequences to use from TinyStories.
        block_size (int): Maximum sequence length for each example.
        enc (tiktoken.Encoding): Tiktoken encoding instance.

    Returns:
        list[list[int]]: List of tokenized TinyStories sequences.
    """
    tinystories_seqs = list()
    if tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        if train_subset_size is not None and isinstance(train_subset_size, int):
            dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    if dataset is not None:
        for sample in tqdm(dataset, total=len(dataset), desc="Tokenizing TinyStories"):
            text = sample["text"]
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    return tinystories_seqs


def _load_input_files(input_files: list[str] | None, block_size: int, enc: tiktoken.Encoding) -> list[list[int]]:
    """Load and tokenize sequences from custom input text files.

    Args:
        input_files (Optional[list[str]]): List of custom text files to include.
        block_size (int): Maximum sequence length for each example.
        enc (tiktoken.Encoding): Tiktoken encoding instance.

    Returns:
        list[list[int]]: List of tokenized sequences from input files.
    """
    seqs = list()
    if input_files:
        for filepath in input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, encoding="utf-8") as f:
                lines = f.readlines()
            for line in tqdm(lines, total=len(lines), desc=f"Tokenizing {filepath}"):
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    seqs.append(tokens)
        print(f"Custom input files: {len(seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    return seqs


def create_train_dataloader(
    tinystories_weight: float,
    train_subset_size: int,
    input_files: list[str] | None,
    block_size: int,
    enc: tiktoken.Encoding,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Create the training dataloader with mixed TinyStories and other text sequences.

    Args:
        tinystories_weight (float): Probability of sampling from TinyStories.
        train_subset_size (int): Number of training sequences to use from TinyStories.
        input_files (Optional[list[str]]): List of custom text files to include.
        block_size (int): Maximum sequence length for each example.
        enc (tiktoken.Encoding): Tiktoken encoding instance.
        batch_size (int): Batch size for training.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the mixed training dataset.
    """
    tinystories_seqs = _load_tinystories(tinystories_weight, train_subset_size, block_size, enc)
    other_seqs = _load_input_files(input_files, block_size, enc)

    p_tiny = tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny > 0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")

    combined_dataset = MixedSequenceDataset(tinystories_seqs=tinystories_seqs, other_seqs=other_seqs, p_tiny=p_tiny)
    return torch.utils.data.DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_seq_collate_fn
    )
