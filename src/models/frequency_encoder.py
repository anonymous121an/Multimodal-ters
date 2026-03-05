import torch
import torch.nn as nn

class FrequencyEncoder(nn.Module):
    """Encodes frequency peaks into a feature vector."""
    def __init__(self, max_freqs=60, embed_dim=128, hidden_dim=256, output_dim=512):
        super().__init__()
        self.max_freqs = max_freqs
        self.embed_dim = embed_dim
        self.freq_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.conv1d = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    def forward(self, frequencies, freq_mask):
        B = frequencies.shape[0]
        freq_expanded = frequencies.unsqueeze(-1)
        freq_embed = self.freq_embed(freq_expanded)
        attn_mask = (freq_mask == 0)
        valid_counts = freq_mask.sum(dim=1)  # (B,)
        if (valid_counts == 0).any():
            freq_attn, _ = self.self_attn(freq_embed, freq_embed, freq_embed, key_padding_mask=None)
            freq_attn = freq_attn * freq_mask.unsqueeze(-1)
        else:
            freq_attn, _ = self.self_attn(freq_embed, freq_embed, freq_embed, key_padding_mask=attn_mask)
        freq_attn = freq_attn + freq_embed
        freq_conv = freq_attn.transpose(1, 2)
        freq_conv = self.conv1d(freq_conv)
        freq_conv = freq_conv * freq_mask.unsqueeze(1)
        freq_global = self.global_pool(freq_conv).squeeze(-1)
        freq_features = self.fc(freq_global)
        return freq_features, freq_conv
