"""Position embedding implementations for Transformer models."""

import inspect

import torch
import torch.nn as nn


class PositionEmbeddingBase(nn.Module):
    """Base helper that enforces a shared API."""

    def apply_positional_embeddings(
        self,
        token_embeddings: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Apply positional information to token embeddings."""
        raise NotImplementedError


class AbsolutePositionEmbedding(PositionEmbeddingBase):
    """Absolute position embedding using learned embeddings."""

    def __init__(self, block_size: int, embed_size: int) -> None:
        """Initialize AbsolutePositionEmbedding.

        Args:
            block_size (int): Maximum sequence length.
            embed_size (int): Embedding dimension.

        Returns:
            None
        """
        super().__init__()
        self.embedding = nn.Embedding(block_size, embed_size)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            position_ids (torch.Tensor): Position indices.

        Returns:
            torch.Tensor: Position embeddings.
        """
        return self.embedding(position_ids)

    def apply_positional_embeddings(
        self,
        token_embeddings: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Add absolute positional embeddings."""
        return token_embeddings + self.forward(position_ids)


class RotaryPositionEmbedding(PositionEmbeddingBase):
    """Rotary position embedding (RoPE) for Transformer models.

    This module applies rotary positional encoding to the last dimension
    of an input tensor (typically queries or keys).

    Expected input shape: (batch_size, seq_len, dim)
    """

    def __init__(self, embed_size: int, base: float = 10000.0) -> None:
        """Initialize RotaryPositionEmbedding.

        Args:
            embed_size (int): Dimension of the embedding (must be even).
            base (float, optional): Base for frequency computation.
                Defaults to 10000.0.

        Returns:
            None
        """
        super().__init__()
        if embed_size % 2 != 0:
            raise ValueError("RotaryPositionEmbedding requires an even 'dim'.")
        self.dim = embed_size
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        # Register as buffer so it moves with .to(device) / .cuda()
        self.register_buffer("inv_freq", inv_freq)

    def _get_cos_sin(
        self,
        position_ids: torch.LongTensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin tables for given positions.

        Args:
            position_ids (torch.LongTensor): Shape (batch, seq_len).
            dtype (torch.dtype): Desired dtype of cos/sin.
            device (torch.device): Device for cos/sin.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                cos, sin each of shape (batch, seq_len, dim // 2).
        """
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)  # (dim//2,)

        # (batch, seq_len, 1) * (1, 1, dim//2) -> (batch, seq_len, dim//2)
        pos = position_ids.to(device=device).unsqueeze(-1).type_as(inv_freq)
        freqs = pos * inv_freq.view(1, 1, -1)

        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        """Apply RoPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            position_ids (torch.LongTensor): Position indices of shape
                (batch, seq_len) or (seq_len,).

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as `x`.
        """
        if x.dim() != 3:
            raise ValueError(
                f"RotaryPositionEmbedding expects x to be 3D (batch, seq_len, dim), got shape {tuple(x.shape)}"
            )

        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Last dimension of x ({dim}) does not match RotaryPositionEmbedding dim ({self.dim}).")

        # Normalize position_ids to shape (batch, seq_len)
        if position_ids.dim() == 1:
            if position_ids.size(0) != seq_len:
                raise ValueError("When position_ids is 1D, its length must match seq_len.")
            # Broadcast same positions across the batch
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.dim() == 2:
            if position_ids.size(0) != batch_size or position_ids.size(1) != seq_len:
                raise ValueError("position_ids must have shape (batch, seq_len) when 2D.")
        else:
            raise ValueError("position_ids must be 1D (seq_len,) or 2D (batch, seq_len).")

        cos, sin = self._get_cos_sin(
            position_ids=position_ids,
            dtype=x.dtype,
            device=x.device,
        )  # (batch, seq_len, dim//2)

        # Reshape x into pairs: (batch, seq_len, dim//2, 2)
        x_reshaped = x.view(batch_size, seq_len, self.dim // 2, 2)
        x_even = x_reshaped[..., 0]  # (batch, seq_len, dim//2)
        x_odd = x_reshaped[..., 1]  # (batch, seq_len, dim//2)

        # Apply rotary transformation
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_odd * cos + x_even * sin

        return torch.stack((rotated_even, rotated_odd), dim=-1).view(batch_size, seq_len, dim)

    def apply_positional_embeddings(
        self,
        token_embeddings: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Apply RoPE to token embeddings."""
        return self.forward(token_embeddings, position_ids)


class NoPositionEmbedding(PositionEmbeddingBase):
    """No position embedding (NoPE).

    This module returns zeros of the appropriate shape so it can be used
    as a drop-in replacement for additive positional embeddings when
    you want to disable them entirely.
    """

    def __init__(self, embed_size: int) -> None:
        """Initialize NoPositionEmbedding.

        Args:
            embed_size (int): Embedding dimension.

        Returns:
            None
        """
        super().__init__()
        self.embed_size = embed_size

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            position_ids (torch.Tensor): Position indices of shape
                (...,) or (..., seq_len).

        Returns:
            torch.Tensor: Zero positional embeddings of shape
                (*position_ids.shape, embed_size).
        """
        shape = position_ids.shape + (self.embed_size,)
        return torch.zeros(shape, device=position_ids.device)

    def apply_positional_embeddings(
        self,
        token_embeddings: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Return embeddings unchanged (no positional signal)."""
        _ = position_ids  # Explicitly acknowledge unused argument
        return token_embeddings


POSITION_EMBEDDING_REGISTRY: dict[str, type[PositionEmbeddingBase]] = {
    "absolute": AbsolutePositionEmbedding,
    "rotary": RotaryPositionEmbedding,
    None: NoPositionEmbedding,
}


def get_position_embedding_cls(name: str) -> type[PositionEmbeddingBase]:
    """Return the positional embedding class registered under `name`."""
    try:
        return POSITION_EMBEDDING_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - better error message
        available = ", ".join(sorted(POSITION_EMBEDDING_REGISTRY))
        raise ValueError(f"Unknown positional embedding '{name}'. Available: {available}") from exc


def _construct_pos_embedding_args(cls: type[PositionEmbeddingBase], kwargs: dict[str, object]) -> dict[str, object]:
    """Filter kwargs to only those accepted by the class initializer."""
    signature = inspect.signature(cls.__init__)
    param_names = set(signature.parameters.keys())
    param_names.discard("self")
    return {k: v for k, v in kwargs.items() if k in param_names}


def build_position_embedding(name: str, **kwargs: object) -> PositionEmbeddingBase:
    """Instantiate a positional embedding by name."""
    embedding_cls = get_position_embedding_cls(name)
    init_kwargs = _construct_pos_embedding_args(embedding_cls, kwargs)
    return embedding_cls(**init_kwargs)
