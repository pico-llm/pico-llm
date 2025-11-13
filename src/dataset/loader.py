"""Data loading utilities for pico-llm."""

import tiktoken
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from .fixed_sequence_dataset import FixedSequenceDataset
from .mixed_sequence_dataset import MixedSequenceDataset


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
    tinystories_weight: float, dataset_subset_size: int, block_size: int, enc: tiktoken.Encoding
) -> list[list[int]]:
    """Load TinyStories dataset from HuggingFace and tokenize sequences.

    Args:
        tinystories_weight (float): Probability of sampling from TinyStories.
        dataset_subset_size (int): Number of sequences to use from TinyStories.
        block_size (int): Maximum sequence length for each example.
        enc (tiktoken.Encoding): Tiktoken encoding instance.

    Returns:
        list[list[int]]: List of tokenized TinyStories sequences.
    """
    tinystories_seqs = list()
    if tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        if dataset_subset_size is not None and isinstance(dataset_subset_size, int):
            dataset = dataset.select(range(dataset_subset_size))
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


def _split_sequences_deterministic(
    sequences: list[list[int]], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Split sequences into train/val/test sets deterministically.

    Args:
        sequences (list[list[int]]): List of token sequences to split.
        train_ratio (float): Ratio of data for training. Default=0.8.
        val_ratio (float): Ratio of data for validation. Default=0.1.
        test_ratio (float): Ratio of data for testing. Default=0.1.

    Returns:
        tuple[list[list[int]], list[list[int]], list[list[int]]]: Train, validation, and test sequences.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    total = len(sequences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_seqs = sequences[:train_end]
    val_seqs = sequences[train_end:val_end]
    test_seqs = sequences[val_end:]

    return train_seqs, val_seqs, test_seqs


def create_dataloaders(
    tinystories_weight: float,
    dataset_subset_size: int,
    input_files: list[str] | None,
    block_size: int,
    enc: tiktoken.Encoding,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test dataloaders with deterministic 80/10/10 split.

    Args:
        tinystories_weight (float): Probability of sampling from TinyStories (used for mixing during loading).
        dataset_subset_size (int): Number of sequences to use from TinyStories.
        input_files (Optional[list[str]]): List of custom text files to include.
        block_size (int): Maximum sequence length for each example.
        enc (tiktoken.Encoding): Tiktoken encoding instance.
        batch_size (int): Batch size for training.
        train_ratio (float): Ratio of data for training. Default=0.8.
        val_ratio (float): Ratio of data for validation. Default=0.1.
        test_ratio (float): Ratio of data for testing. Default=0.1.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
    """
    # Load all sequences
    tinystories_seqs = _load_tinystories(tinystories_weight, dataset_subset_size, block_size, enc)
    other_seqs = _load_input_files(input_files, block_size, enc)

    # Combine all sequences deterministically (first TinyStories, then other)
    all_sequences = tinystories_seqs + other_seqs

    if len(all_sequences) == 0:
        raise ValueError("No data found! Both TinyStories and other sets are empty.")

    print(f"Total sequences loaded: {len(all_sequences)}")
    print(f"  - TinyStories: {len(tinystories_seqs)}")
    print(f"  - Custom files: {len(other_seqs)}")

    # Split deterministically
    train_seqs, val_seqs, test_seqs = _split_sequences_deterministic(
        all_sequences, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )

    print(f"Data split (deterministic):")
    print(f"  - Train: {len(train_seqs)} ({len(train_seqs)/len(all_sequences)*100:.1f}%)")
    print(f"  - Val: {len(val_seqs)} ({len(val_seqs)/len(all_sequences)*100:.1f}%)")
    print(f"  - Test: {len(test_seqs)} ({len(test_seqs)/len(all_sequences)*100:.1f}%)")

    # Create datasets
    train_dataset = FixedSequenceDataset(train_seqs)
    val_dataset = FixedSequenceDataset(val_seqs)
    test_dataset = FixedSequenceDataset(test_seqs)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_seq_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_seq_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_seq_collate_fn
    )

    return train_loader, val_loader, test_loader


def create_train_dataloader(
    tinystories_weight: float,
    train_subset_size: int,
    input_files: list[str] | None,
    block_size: int,
    enc: tiktoken.Encoding,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Create the training dataloader with mixed TinyStories and other text sequences.

    This function is kept for backward compatibility but is deprecated.
    Use create_dataloaders() instead for train/val/test splits.

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
