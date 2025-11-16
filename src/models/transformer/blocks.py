"""Transformer building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class RMSNorm(nn.Module):
    """RMSNorm implementation."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initialize RMSNorm.

        Args:
            dim (int): Dimension of the input.
            eps (float): Small value to avoid division by zero.

        Returns:
            None
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class FeedForward(nn.Module):
    """Position-wise feed-forward network (GPT-2 style)."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        """Initialize FeedForward.

        The inner dimension is set to 4 times the model dimension, following GPT-2 architecture.

        Args:
            d_model (int): Dimension of the model.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """GPT-2 style transformer block with configurable normalization."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, norm: str = "prenorm") -> None:
        """Initialize TransformerBlock.

        Args:
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            norm (str): Normalization type ('prenorm' or 'postnorm').
        """
        super().__init__()
        self.norm = norm
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device, dtype=x.dtype)
        if self.norm == "prenorm":
            # Pre-norm: norm before attention/FF (LLaMA style)
            x_norm = self.norm1(x)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
            x = x + self.dropout(attn_out)
            x = x + self.ff(self.norm2(x))
        else:
            # Post-norm: norm after attention/FF (original Transformer)
            attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
            x = self.norm1(x + self.dropout(attn_out))
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)

        return x
