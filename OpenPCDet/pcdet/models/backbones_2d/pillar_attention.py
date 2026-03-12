import math

import torch
import torch.nn as nn


class PillarAttention(nn.Module):
    """Self-attention block for sparse pillar tokens.

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
