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


class SpectralCNNEncoder(nn.Module):
    """
    1D CNN encoder for binned Raman/TERS spectra.

    Treats the binned spectrum as a 1D signal and applies hierarchical
    convolutions to extract local peak features, following the approach of
    Acquarelli et al. (Analytica Chimica Acta, 2017) and Zhang et al.
    (Analytica Chimica Acta, 2018).

    Returns:
        freq_features : (B, output_dim)     — global descriptor (AdaptiveMaxPool)
        freq_seq      : (B, hidden_dim, T)  — local feature map usable as
                        per-token sequence for cross-modal attention, analogous
                        to how DETR (Carion et al., ECCV 2020) feeds CNN feature
                        maps into a transformer encoder.
    """
    def __init__(self, num_bins=100, hidden_dim=256, output_dim=512):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1: local peak detection (~40 cm-1 receptive field for 100-bin input)
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                          # 100 -> 50

            # Block 2: mid-scale spectral features
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),                          # 50 -> 25

            # Block 3: spectral-region features (each of 25 tokens ~ 160 cm-1 window)
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, frequencies, freq_mask=None):
        # freq_mask unused for binning (no padding); accepted for API compatibility
        x = frequencies.unsqueeze(1)            # (B, 1, num_bins)
        freq_seq = self.conv_blocks(x)          # (B, hidden_dim, T)  T=25 for 100 bins
        freq_global = self.global_pool(freq_seq).squeeze(-1)  # (B, hidden_dim)
        freq_features = self.fc(freq_global)    # (B, output_dim)
        return freq_features, freq_seq
