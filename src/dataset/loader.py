"""Data loading utilities for pico-llm."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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


def _load_tinystories(block_size: int, enc: tiktoken.Encoding) -> list[list[int]]:
    """Load TinyStories dataset from HuggingFace and tokenize sequences.

    Args:
        block_size (int): Maximum sequence length for each example.
        enc (tiktoken.Encoding): Tiktoken encoding instance.

    Returns:
        list[list[int]]: List of tokenized TinyStories sequences.
    """
    tinystories_seqs = list()
    print("Loading TinyStories from huggingface...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    for sample in tqdm(dataset, total=len(dataset), desc="Tokenizing TinyStories"):
        text = sample["text"]
        tokens = enc.encode(text)
        tokens = tokens[:block_size]
        if len(tokens) > 0:
            tinystories_seqs.append(tokens)
    print(f"TinyStories sequences: {len(tinystories_seqs)}")

    return tinystories_seqs


def _tokenize_sample(sample: dict, block_size: int, encoding_name: str) -> list[int] | None:
    """Tokenize a single sample from the dataset.

    Args:
        sample (dict): Dataset sample containing 'text' field.
        block_size (int): Maximum sequence length.
        encoding_name (str): Name of the tiktoken encoding to use.

    Returns:
        list[int] | None: Tokenized sequence or None if empty.
    """
    # Create encoding in worker process (tiktoken.Encoding is not picklable)
    enc = tiktoken.get_encoding(encoding_name)
    text = sample["text"]
    tokens = enc.encode(text)
    tokens = tokens[:block_size]
    return tokens if len(tokens) > 0 else None


def _load_tinystories_parallel(block_size: int, enc: tiktoken.Encoding) -> list[list[int]]:
    """Load TinyStories dataset from HuggingFace and tokenize sequences using parallel processing.

    Args:
        block_size (int): Maximum sequence length for each example.
        enc (tiktoken.Encoding): Tiktoken encoding instance.

    Returns:
        list[list[int]]: List of tokenized TinyStories sequences (same order as dataset).
    """
    print("Loading TinyStories from huggingface...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # Get encoding name to pass to workers (tiktoken.Encoding objects are not picklable)
    encoding_name = enc.name

    # Determine number of workers
    num_workers = multiprocessing.cpu_count() or 1
    print(f"Using {num_workers} workers for parallel tokenization...")

    # Create partial function with fixed parameters
    tokenize_func = partial(_tokenize_sample, block_size=block_size, encoding_name=encoding_name)

    tinystories_seqs = list()

    # Process in parallel while maintaining order
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use map to maintain order and chunksize for efficiency
        chunksize = max(1, len(dataset) // (num_workers * 4))
        results = list(
            tqdm(
                executor.map(tokenize_func, dataset, chunksize=chunksize),
                total=len(dataset),
                desc="Tokenizing TinyStories",
            )
        )

    # Filter out None values (empty sequences)
    tinystories_seqs = [tokens for tokens in results if tokens is not None]

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


def create_dataloaders(
    dataset_subset_size: int | None,
    input_files: list[str] | None,
    block_size: int,
    enc: tiktoken.Encoding,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    dataset_type: str = "fixed",
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        dataset_subset_size (int | None): Number of sequences to use from the dataset. If None, use all data.
        input_files (Optional[list[str]]): List of custom text files to include.
        block_size (int): Maximum sequence length for each example.
        enc (tiktoken.Encoding): Tiktoken encoding instance.
        batch_size (int): Batch size for training.
        train_ratio (float): Ratio of data for training.
        val_ratio (float): Ratio of data for validation.
        dataset_type (str): Type of dataset to use ("fixed" or "mixed").
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
    """
    # load all sequences
    # tinystories_seqs = _load_tinystories(block_size, enc)
    tinystories_seqs = _load_tinystories_parallel(block_size, enc)
    other_seqs = _load_input_files(input_files, block_size, enc)

    # combine all sequences
    all_sequences = tinystories_seqs + other_seqs
    all_sequences = all_sequences[:dataset_subset_size] if dataset_subset_size else all_sequences
    num_sequences = len(all_sequences)

    if num_sequences == 0:
        raise ValueError("No data found! Both TinyStories and other sets are empty.")

    print(f"Total sequences loaded: {num_sequences}")
    print(f"  - TinyStories: {len(tinystories_seqs)}")
    print(f"  - Custom files: {len(other_seqs)}")

    # Split deterministically
    if train_ratio == 0.0:
        raise ValueError("Train ratio must be greater than 0.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("Train + Val ratio must be less than 1.")

    train_len = int(num_sequences * train_ratio)
    val_len = int(num_sequences * val_ratio)
    shuffled_indices = torch.randperm(num_sequences).tolist()
    train_seqs = [all_sequences[i] for i in shuffled_indices[:train_len]]
    val_seqs = [all_sequences[i] for i in shuffled_indices[train_len : train_len + val_len]]
    test_seqs = [all_sequences[i] for i in shuffled_indices[train_len + val_len :]]

    print("Data splits:")
    print(f"  - Train: {len(train_seqs)} ({len(train_seqs) / num_sequences * 100:.1f}%)")
    print(f"  - Val: {len(val_seqs)} ({len(val_seqs) / num_sequences * 100:.1f}%)")
    print(f"  - Test: {len(test_seqs)} ({len(test_seqs) / num_sequences * 100:.1f}%)")

    # create datasets
    dataset_cls = FixedSequenceDataset if dataset_type == "fixed" else MixedSequenceDataset
    train_dataset = dataset_cls(train_seqs)
    val_dataset = dataset_cls(val_seqs)
    test_dataset = dataset_cls(test_seqs)

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=_seq_collate_fn,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=_seq_collate_fn,
        )
        if len(val_dataset) > 0
        else None
    )
    test_loader = (
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=_seq_collate_fn,
        )
        if len(test_dataset) > 0
        else None
    )

    return train_loader, val_loader, test_loader
