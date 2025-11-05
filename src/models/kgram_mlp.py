"""Implementation of a k-gram MLP model for text generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class KGramMLPSeqModel(nn.Module):
    """Implementation of a k-gram MLP-based sequence model.

    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(
        self, vocab_size: int, k: int = 3, embed_size: int = 1024, num_inner_layers: int = 1, chunk_size: int = 1
    ) -> "KGramMLPSeqModel":
        """Initialize the k-gram MLP-based sequence model.

        Args:
            vocab_size (int): Size of the vocabulary.
            k (int): Number of previous tokens to consider.
            embed_size (int): Dimension of the embedding layer.
            num_inner_layers (int): Number of inner layers in the MLP.
            chunk_size (int): Number of time steps to process in one chunk.

        Returns:
            KGramMLPSeqModel: An instance of the k-gram MLP sequence model.
        """
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # fill in

        self.net = None

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

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size,
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        return torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
