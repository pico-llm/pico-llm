"""Implementation of a k-gram MLP model for text generation."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import PyTorchModelHubMixin


class KGramMLPSeqModel(nn.Module, PyTorchModelHubMixin):
    """Implementation of a k-gram MLP-based sequence model.

    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Note: We are going to override behavior by using Embedding layers instead of one-hot encodings.
    """

    def __init__(
        self,
        vocab_size: int,
        k: int = 3,
        embed_size: int = 1024,
        num_inner_layers: int = 1,
        embedding_type: Literal["full", "scaled", "onehot"] = "full",
    ) -> "KGramMLPSeqModel":
        """Initialize the k-gram MLP-based sequence model.

        Args:
            vocab_size (int): Size of the vocabulary.
            k (int): Number of previous tokens to consider.
            embed_size (int): Dimension of the embedding layer.
            num_inner_layers (int): Number of inner layers in the MLP.
            embedding_type (str): Mode of input representation. Options are "full", "scaled", "onehot".

        Returns:
            KGramMLPSeqModel: An instance of the k-gram MLP sequence model.
        """
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.embedding_type = embedding_type

        if self.embedding_type == "onehot":
            self.embedding = None
            input_dim = k * vocab_size
        elif self.embedding_type == "full":
            self.embedding = nn.Embedding(vocab_size, embed_size)
            input_dim = k * embed_size
        elif self.embedding_type == "scaled":
            self.embedding = nn.Embedding(vocab_size, embed_size // k)
            input_dim = k * (embed_size // k)
        else:
            raise NotImplementedError(f"Embedding type '{self.embedding_type}' is not implemented.")

        # output dim is vocab size for logits
        output_dim = vocab_size

        # build the MLP
        layers = list()
        layers.append(nn.Linear(input_dim, embed_size))
        layers.append(nn.SiLU())
        for _ in range(num_inner_layers):
            layers.append(nn.Linear(embed_size, embed_size))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(embed_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass of the k-gram MLP model.

        Args:
            tokens_seq (torch.Tensor): Input token sequence of shape (seq_len, batch).

        Returns:
            torch.Tensor: Output logits of shape (seq_len, batch, vocab_size).
        """
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device

        # prepend padding tokens for positions < k
        # shape: (k + seq_len, batch)
        padded_tokens = torch.cat([torch.zeros(self.k, batch_size, dtype=torch.long, device=device), tokens_seq], dim=0)

        # create index tensor for gathering k-grams
        # for each position t in original sequence, we want the last k tokens before position t
        # original position t maps to padded position t+k (since we prepended k zeros)
        # we want padded positions [t, t+k) which gives us the k tokens ending just before position t
        # create indices: for each t, we want positions [t, t+1, ..., t+k-1] in padded
        t_indices = torch.arange(seq_len, device=device)[:, None]  # (seq_len, 1)
        k_offsets = torch.arange(self.k, device=device)[None, :]  # (1, k)
        indices = (t_indices + k_offsets)[:, None, :]  # (seq_len, 1, k)

        # gather k-grams: (seq_len, k, batch) -> transpose to (seq_len, batch, k)
        context_tokens = padded_tokens[indices].squeeze(1).transpose(1, 2)  # (seq_len, batch, k)

        # flatten for processing: (seq_len * batch, k)
        context_flat = context_tokens.reshape(seq_len * batch_size, self.k)

        # apply embeddings or one-hot encoding
        if self.embedding is None:
            # one-hot mode: (seq_len * batch, k, vocab_size)
            context_embeddings = F.one_hot(context_flat, num_classes=self.vocab_size).float()
        else:
            # embedding mode: (seq_len * batch, k, embed_size)
            context_embeddings = self.embedding(context_flat)

        # flatten k-gram dimensions: (seq_len * batch, k * dim)
        # where dim is vocab_size (one-hot) or embed_size (embedding)
        context_input = context_embeddings.reshape(seq_len * batch_size, -1)

        # process through MLP: (seq_len * batch, vocab_size)
        logits_flat = self.net(context_input)

        # reshape back to (seq_len, batch, vocab_size)
        return logits_flat.reshape(seq_len, batch_size, self.vocab_size)
