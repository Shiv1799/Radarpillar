import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import torch
import torch.nn as nn


def extract_pillar_tokens(spatial_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract non-empty pillar tokens from a BEV feature map.

    Args:
        spatial_features: BEV tensor with shape (B, C, H, W).

    Returns:
        tokens: Tensor with shape (p, C), where p is number of occupied pillars.
        indices: Flat indices with shape (p,) over the (B * H * W) grid.
    """
    if spatial_features.ndim != 4:
        raise ValueError(
            f"extract_pillar_tokens expects input shape (B, C, H, W), got {tuple(spatial_features.shape)}"
        )

    batch_size, channels, height, width = spatial_features.shape

    # (B, C, H, W) -> (B, C, H*W)
    flat_bev = spatial_features.reshape(batch_size, channels, height * width)
    # (B, C, H*W) -> (B, H*W, C)
    flat_tokens = flat_bev.transpose(1, 2).contiguous()

    # Occupied if token feature magnitude is non-zero.
    occupied_mask = flat_tokens.abs().sum(dim=-1) > 0  # (B, H*W)

    # Flatten batch + spatial dimensions so step-4 style scatter can restore with (B, H, W).
    flat_tokens_all = flat_tokens.reshape(batch_size * height * width, channels)
    occupied_indices = occupied_mask.reshape(batch_size * height * width).nonzero(as_tuple=False).squeeze(1)

    tokens = flat_tokens_all[occupied_indices]
    return tokens, occupied_indices


def scatter_pillar_tokens(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    grid_size: Tuple[int, int],
    batch_size: int,
    channels: int,
) -> torch.Tensor:
    """Scatter sparse pillar tokens back to a dense BEV tensor.

    Args:
        tokens: Tensor of shape (p, C).
        indices: Flat indices of shape (p,) over (B * H * W).
        grid_size: Tuple (H, W).
        batch_size: B.
        channels: C.

    Returns:
        Dense BEV tensor with shape (B, C, H, W).
    """
    if tokens.ndim != 2:
        raise ValueError(f"scatter_pillar_tokens expects tokens shape (p, C), got {tuple(tokens.shape)}")
    if indices.ndim != 1:
        raise ValueError(f"scatter_pillar_tokens expects indices shape (p,), got {tuple(indices.shape)}")

    height, width = grid_size
    bev_flat = torch.zeros(
        batch_size, channels, height * width, dtype=tokens.dtype, device=tokens.device
    )

    if indices.numel() > 0:
        flat_all = bev_flat.permute(0, 2, 1).contiguous().view(batch_size * height * width, channels)
        flat_all[indices.long()] = tokens

    return bev_flat.view(batch_size, channels, height, width)


class PillarAttention(nn.Module):
    """Self-attention block for sparse pillar tokens.

    Supports:
      - a single token tensor: (N, C)
      - a list of per-batch token tensors: [(p_0, C), (p_1, C), ...]
        where p_i can vary per batch element.
    Input shape:
        X: (N, C)
    Output shape:
        (N, C)
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"PillarAttention expects input shape (N, C), got {tuple(x.shape)}")

        q = self.q_proj(x)  # (N, C)
        k = self.k_proj(x)  # (N, C)
        v = self.v_proj(x)  # (N, C)

        scale = 1.0 / math.sqrt(self.feature_dim)
        attn_logits = torch.matmul(q, k.transpose(0, 1)) * scale  # (N, N)
        attn = torch.softmax(attn_logits, dim=-1)  # (N, N)
        out = torch.matmul(attn, v)  # (N, C)
        return out
