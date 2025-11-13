"""Implementation of a Transformer model for text generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin



class RMSNorm(nn.Module):
    """RMSNorm implementation."""

    def __init__(self, dim: int, eps: float = 1e-5) -> "RMSNorm":
        """Initialize RMSNorm.

        Args:
            dim (int): Dimension of the input.
            eps (float): Small value to avoid division by zero.

        Returns:
            RMSNorm: An instance of RMSNorm.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class FeedForward(nn.Module):
    """Position-wise feed-forward network (GPT-2 style)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        """Initialize FeedForward.

        Args:
            d_model (int): Dimension of the model.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """GPT-2 style transformer block with configurable normalization."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, pre_norm: bool = True):
        """Initialize TransformerBlock.

        Args:
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            pre_norm (bool): If True, use pre-norm. If False, use post-norm.
        """
        super().__init__()
        self.pre_norm = pre_norm
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, dropout)
        self.norm1 = RMSNorm(d_model) # changed from LayerNorm to RMSNorm
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor, optional): Attention mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.pre_norm:
            # Pre-norm: norm before attention/FF (GPT-2, LLaMA style)
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), 
                                     attn_mask=mask, need_weights=False)
            x = x + self.dropout(attn_out)
            x = x + self.ff(self.norm2(x))
        else:
            # Post-norm: norm after attention/FF (original Transformer)
            attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.ff(x))
        
        return x


class TransformerModel(nn.Module, PyTorchModelHubMixin):
    """Transformer model for text generation."""

    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        n_heads: int, 
        n_blocks: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pre_norm: bool = True
    ) -> "TransformerModel":
        """Initialize TransformerModel.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            n_blocks (int): Number of transformer blocks.
            max_seq_len (int): Maximum sequence length for positional embeddings.
            dropout (float): Dropout rate.
            pre_norm (bool): If True, use pre-norm (GPT-2 style). 
                           If False, use post-norm (original Transformer).

        Returns:
            TransformerModel: An instance of TransformerModel.
        """
        super().__init__()
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout, pre_norm=pre_norm) 
            for _ in range(n_blocks)
        ])
        
        # Final norm
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying 
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights (GPT-2 style)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        """Forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch, seq_len)
            
        Returns:
            torch.Tensor: Output logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask for autoregressive generation
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final norm (always apply for pre-norm, optional for post-norm)
        if self.pre_norm:
            x = self.norm(x)
        
        logits = self.lm_head(x)
        
        return logits