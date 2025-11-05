"""MixedSequenceDataset class and data loading utilities for pico-llm."""

import random

import torch


class MixedSequenceDataset(torch.utils.data.Dataset):
    """Dataset that mixes TinyStories sequences with other text sequences.

    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """

    def __init__(
        self, tinystories_seqs: list[list[int]], other_seqs: list[list[int]], p_tiny: float
    ) -> "MixedSequenceDataset":
        """Initialize the MixedSequenceDataset.

        Args:
            tinystories_seqs (list[list[int]]): List of TinyStories token sequences.
            other_seqs (list[list[int]]): List of other text token sequences.
            p_tiny (float): Probability of sampling from TinyStories.
        """
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = len(self.tinystories_seqs) > 0
        self.has_other = len(self.other_seqs) > 0

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self) -> int:
        """Return the total number of sequences in the dataset."""
        return self.total_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a random sequence from either TinyStories or other sequences.

        Args:
            idx (int): Index (not used since sampling is random).

        Returns:
            torch.Tensor: 1D LongTensor of the selected token sequence.
        """
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)
