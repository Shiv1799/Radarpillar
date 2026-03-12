import torch
import torch.nn as nn

from .pillar_attention import PillarAttention, extract_pillar_tokens, scatter_pillar_tokens


class RadarPillarAttentionBlock(nn.Module):
    """Complete radar pillar attention block.

    Pipeline:
        1) extract non-empty pillar tokens from BEV grid
        2) apply PillarAttention on sparse tokens
        3) scatter tokens back into dense BEV grid

    Input:
        spatial_features: (B, C, H, W)

    Output:
        spatial_features: (B, C, H, W)
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = PillarAttention(feature_dim)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        if spatial_features.ndim != 4:
            raise ValueError(
                "RadarPillarAttentionBlock expects input shape (B, C, H, W), "
                f"got {tuple(spatial_features.shape)}"
            )

        batch_size, channels, height, width = spatial_features.shape
        tokens, indices = extract_pillar_tokens(spatial_features)

        if tokens.numel() > 0:
            tokens = self.attention(tokens)

        spatial_features = scatter_pillar_tokens(
            tokens=tokens,
            indices=indices,
            grid_size=(height, width),
            batch_size=batch_size,
            channels=channels,
        )
        return spatial_features
