"""LSTM model implementation for text generation."""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class LSTMSeqModel(nn.Module, PyTorchModelHubMixin):
    """Implementation of an LSTM-based sequence model."""

    def __init__(self, vocab_size: int, embed_size: int = 1024, hidden_size: int = 1024) -> "LSTMSeqModel":
        """Initialize the LSTM-based sequence model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimension of the embedding layer.
            hidden_size (int): Dimension of the LSTM hidden state.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LSTM model.

        Args:
            tokens_seq (torch.Tensor): Input token sequence of shape (seq_len, batch_size).

        Returns:
            torch.Tensor: Logits of shape (seq_len, batch_size, vocab_size).
        """
        emb = self.embedding(tokens_seq)  # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)  # (seq_len, batch, hidden)
        return self.linear(out)  # (seq_len, batch, vocab_size)
