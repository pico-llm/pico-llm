"""Implementation of a Transformer model for text generation."""

from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .blocks import RMSNorm, TransformerBlock
from .position_embeddings import build_position_embedding


class TransformerModel(nn.Module, PyTorchModelHubMixin):
    """Transformer model for text generation."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        n_heads: int,
        n_blocks: int,
        pos_embed_type: str = "abs",
        block_size: int = 1024,
        norm: str = "prenorm",
        dropout: float = 0.1,
    ) -> None:
        """Initialize TransformerModel.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Embedding dimension (d_model).
            n_heads (int): Number of attention heads.
            n_blocks (int): Number of transformer blocks.
            pos_embed_type (str): Type of positional embeddings. Options are "abs" (absolute), "rope" (RoPE).
            block_size (int): Maximum sequence length for positional embeddings.
            norm (str): Normalization style ('prenorm' or 'postnorm').
            dropout (float): Dropout rate.

        Returns:
            None
        """
        super().__init__()

        self.d_model = embed_size
        self.norm_type = norm
        self.pos_embed_type = pos_embed_type

        # Map main's parameter names to pos_embed's implementation names
        pos_embed_mapping = {
            "abs": "absolute",
            "rope": "rotary",
        }
        pos_embedding_type = pos_embed_mapping.get(pos_embed_type, "absolute")

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        embedding_kwargs: dict[str, Any] = {
            "block_size": block_size,
            "embed_size": embed_size,
        }
        self.position_embedding = build_position_embedding(pos_embedding_type, **embedding_kwargs)
        self.dropout = nn.Dropout(dropout)

        # Cache position IDs as a buffer (not a parameter, won't be trained)
        self.register_buffer("position_ids", torch.arange(block_size), persistent=False)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_size, n_heads, dropout=dropout, norm=self.norm_type) for _ in range(n_blocks)]
        )

        # Final norm
        self.norm = RMSNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights (GPT-2 style)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch, seq_len)

        Returns:
            torch.Tensor: Output logits of shape (batch, seq_len, vocab_size)
        """
        _, seq_len = input_ids.shape

        # Embeddings
        position_ids = self.position_ids[:seq_len]
        token_embeddings = self.token_embedding(input_ids)
        x = self.position_embedding.apply_positional_embeddings(token_embeddings, position_ids)
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm (always apply for pre-norm, optional for post-norm)
        if self.norm_type == "prenorm":
            x = self.norm(x)

        return self.lm_head(x)
