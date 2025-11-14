"""FixedSequenceDataset class for deterministic data access."""

import torch


class FixedSequenceDataset(torch.utils.data.Dataset):
    """Dataset that provides fixed-index access to a list of sequences.

    This dataset uses fixed indices for deterministic data access, making it suitable for train/val/test splits.
    """

    def __init__(self, sequences: list[list[int]]) -> "FixedSequenceDataset":
        """Initialize the FixedSequenceDataset.

        Args:
            sequences (list[list[int]]): List of token sequences.
        """
        super().__init__()
        self.sequences = sequences
        if len(self.sequences) == 0:
            raise ValueError("No sequences provided to FixedSequenceDataset.")

    def __len__(self) -> int:
        """Return the total number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sequence by index.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            torch.Tensor: 1D LongTensor of the token sequence.
        """
        return torch.tensor(self.sequences[idx], dtype=torch.long)
