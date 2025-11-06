"""Implementation of a k-gram MLP model for text generation."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class KGramMLPSeqModel(nn.Module):
    """Implementation of a k-gram MLP-based sequence model.

    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.

    Note: We are going to override behavior by using Embedding layers instead of one-hot encodings.
    """

    def __init__(
        self,
        vocab_size: int,
        k: int = 3,
        embed_size: int = 1024,
        num_inner_layers: int = 1,
        chunk_size: int = 1,
        embedding_type: Literal["full", "scaled", "onehot"] = "full",
    ) -> "KGramMLPSeqModel":
        """Initialize the k-gram MLP-based sequence model.

        Args:
            vocab_size (int): Size of the vocabulary.
            k (int): Number of previous tokens to consider.
            embed_size (int): Dimension of the embedding layer.
            num_inner_layers (int): Number of inner layers in the MLP.
            chunk_size (int): Number of time steps to process in one chunk.
            embedding_type (str): Mode of input representation. Options are "full", "scaled", "onehot".

        Returns:
            KGramMLPSeqModel: An instance of the k-gram MLP sequence model.
        """
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size
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
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0] * needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t - self.k : t, b].tolist()

                    # convert context_ids to embeddings or one-hot
                    context_tensor = torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device)
                    if self.embedding is None:
                        # one-hot mode
                        context_embeddings = F.one_hot(
                            context_tensor, num_classes=self.vocab_size
                        ).float()  # (k, vocab_size)
                    else:
                        # embedding mode
                        context_embeddings = self.embedding(context_tensor)  # (k, embed_size)
                    context_flat = context_embeddings.flatten().unsqueeze(0)  # (1, k * dim)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        return torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
