import torch
import torch.nn as nn

from .pillar_attention import PillarAttention, extract_pillar_tokens, scatter_pillar_tokens


class RadarPillarAttentionBlock(nn.Module):
    """Radar pillar transformer block on sparse BEV pillar tokens.

    Token pipeline:
        X = X + MLP1(X)
        X = X + PillarAttention(X)
        X = X + MLP2(X)

    The transformer operations are performed only on occupied pillar tokens.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, feature_dim),
        )
        self.attention = PillarAttention(feature_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, feature_dim),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        if spatial_features.ndim != 4:
            raise ValueError(
                "RadarPillarAttentionBlock expects input shape (B, C, H, W), "
                f"got {tuple(spatial_features.shape)}"
            )

        batch_size, channels, height, width = spatial_features.shape
        tokens, indices = extract_pillar_tokens(spatial_features)

        if tokens.numel() > 0:
            tokens = tokens + self.mlp1(tokens)
            tokens = tokens + self.attention(tokens)
            tokens = tokens + self.mlp2(tokens)

        spatial_features = scatter_pillar_tokens(
            tokens=tokens,
            indices=indices,
            grid_size=(height, width),
            batch_size=batch_size,
            channels=channels,
        )
        return spatial_features
