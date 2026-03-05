import torch
import torch.nn as nn

class EarlyFusion(nn.Module):
    """Early Fusion: Project frequency features to spatial map."""
    def __init__(self, freq_dim, spatial_size=256, out_channels=64):
        super().__init__()
        self.spatial_size = spatial_size
        self.freq_to_spatial = nn.Sequential(
            nn.Linear(freq_dim, spatial_size * spatial_size // 16),
            nn.ReLU(),
            nn.Linear(spatial_size * spatial_size // 16, spatial_size * spatial_size)
        )
        self.combine_conv = nn.Conv2d(1, out_channels, kernel_size=1)
    def forward(self, image, freq_features):
        B = image.shape[0]
        freq_spatial = self.freq_to_spatial(freq_features)
        freq_spatial = freq_spatial.view(B, 1, self.spatial_size, self.spatial_size)
        freq_spatial = self.combine_conv(freq_spatial)
        fused = torch.cat([image, freq_spatial], dim=1)
        return fused

class LateFusion(nn.Module):
    """Late Fusion: Combine at bottleneck."""
    def __init__(self, image_channels, freq_dim, output_channels):
        super().__init__()
        self.freq_project = nn.Sequential(
            nn.Linear(freq_dim, output_channels),
            nn.ReLU()
        )
        self.combine = nn.Sequential(
            nn.Conv2d(image_channels + output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    def forward(self, image_bottleneck, freq_features):
        B, C, H, W = image_bottleneck.shape
        freq_proj = self.freq_project(freq_features)
        freq_spatial = freq_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined = torch.cat([image_bottleneck, freq_spatial], dim=1)
        fused = self.combine(combined)
        return fused

# Legacy cross-modal attention (kept for reference).
# class CrossModalAttention(nn.Module):
#     """Cross-Modal Attention: Use frequency to attend to image features."""
#     def __init__(self, image_channels, freq_hidden_dim, attention_dim=64):
#         super().__init__()
#         self.image_proj = nn.Conv2d(image_channels, attention_dim, kernel_size=1)
#         self.freq_proj = nn.Linear(freq_hidden_dim, attention_dim)
#         self.attention = nn.Sequential(
#             nn.Conv2d(attention_dim, attention_dim, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(attention_dim, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#     def forward(self, image_features, freq_seq, freq_mask):
#         B, C, H, W = image_features.shape
#         img_proj = self.image_proj(image_features)
#         freq_global = (freq_seq * freq_mask.unsqueeze(1)).sum(dim=2) / freq_mask.sum(dim=1, keepdim=True).clamp(min=1)
#         freq_proj = self.freq_proj(freq_global)
#         freq_spatial = freq_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
#         combined = img_proj + freq_spatial
#         attention_weights = self.attention(combined)
#         attended = image_features * attention_weights
#         return attended, attention_weights

class CrossModalAttention(nn.Module):
    """Transformer-style cross-attention (Vaswani et al., 2017)."""
    def __init__(self, image_channels, freq_hidden_dim, attention_dim=64, num_heads=4):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})")

        self.query_proj = nn.Conv2d(image_channels, attention_dim, kernel_size=1)
        self.key_proj = nn.Linear(freq_hidden_dim, attention_dim)
        self.value_proj = nn.Linear(freq_hidden_dim, attention_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(attention_dim)
        self.out_proj = nn.Conv2d(attention_dim, image_channels, kernel_size=1)

    def forward(self, image_features, freq_seq, freq_mask):
        B, C, H, W = image_features.shape

        # Q: image tokens, K/V: frequency tokens.
        query = self.query_proj(image_features).flatten(2).transpose(1, 2)  # (B, H*W, D)
        freq_tokens = freq_seq.transpose(1, 2)  # (B, L, F)
        key = self.key_proj(freq_tokens)        # (B, L, D)
        value = self.value_proj(freq_tokens)    # (B, L, D)

        key_padding_mask = None
        if freq_mask is not None:
            key_padding_mask = (freq_mask == 0)

        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )

        # Residual + normalization in token space.
        fused_tokens = self.norm(query + attn_out)           # (B, H*W, D)
        fused_map = fused_tokens.transpose(1, 2).reshape(B, -1, H, W)
        attended = image_features + self.out_proj(fused_map)
        return attended, attn_weights

class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""
    def __init__(self, freq_dim, num_channels):
        super().__init__()
        self.gamma = nn.Linear(freq_dim, num_channels)
        self.beta = nn.Linear(freq_dim, num_channels)
    def forward(self, image_features, freq_features):
        gamma = self.gamma(freq_features).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(freq_features).unsqueeze(-1).unsqueeze(-1)
        return gamma * image_features + beta
