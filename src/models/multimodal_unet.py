import torch
import torch.nn as nn
from src.models.layers import ResBlock, AttentionBlock2d
from src.fusion_techniques import EarlyFusion, LateFusion, CrossModalAttention, FiLM
from .frequency_encoder import FrequencyEncoder, SpectralCNNEncoder

class MultimodalAttentionUNet(nn.Module):
    """
    Multimodal Attention U-Net with configurable fusion strategy.
    Fusion types: 'none', 'early', 'late', 'attention', 'film', 'hybrid'
    """
    def __init__(self, 
                 in_channels=400, 
                 out_channels=4,
                 filters=[64, 128, 256, 512],
                 att_channels=64,
                 kernel_size=[3, 3, 3, 3],
                 max_freqs=60,
                 freq_embed_dim=128,
                 freq_output_dim=512,
                 fusion_type='late',
                 freq_encoder_type='transformer',
                 num_bins=400,
                 bottleneck_dropout_p=0.0):
        super().__init__()
        self.fusion_type = fusion_type
        self.filters = filters
        if fusion_type != 'none':
            if freq_encoder_type == 'spectral_cnn':
                self.freq_encoder = SpectralCNNEncoder(
                    num_bins=num_bins,
                    hidden_dim=256,
                    output_dim=freq_output_dim,
                )
            else:  # 'transformer' (default)
                self.freq_encoder = FrequencyEncoder(
                    max_freqs=max_freqs,
                    embed_dim=freq_embed_dim,
                    hidden_dim=256,
                    output_dim=freq_output_dim,
                )
        if fusion_type == 'early':
            self.early_fusion = EarlyFusion(freq_output_dim, spatial_size=256, out_channels=64)
            in_channels = in_channels + 64
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1)
        enc_in_channels = 64
        self.encoder = nn.ModuleList()
        for f, k in zip(filters, kernel_size):
            self.encoder.append(ResBlock(enc_in_channels, f, k))
            enc_in_channels = f
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if fusion_type == 'late':
            self.late_fusion = LateFusion(filters[-1], freq_output_dim, filters[-1])
        self.bottom = ResBlock(filters[-1], filters[-1], kernel_size[-1])
        self.bottleneck_dropout = nn.Dropout2d(p=bottleneck_dropout_p)
        if fusion_type in ['attention', 'hybrid']:
            self.cross_attentions = nn.ModuleList()
            for f in filters[::-1]:
                self.cross_attentions.append(CrossModalAttention(f, 256, attention_dim=64))
        if fusion_type in ['film', 'hybrid']:
            self.film_layers = nn.ModuleList()
            for f in filters[::-1]:
                self.film_layers.append(FiLM(freq_output_dim, f))
        self.attentions = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        prev_channel = filters[-1]
        decoder_filters = filters[::-1]
        decoder_kernels = kernel_size[::-1]
        for f, k in zip(decoder_filters, decoder_kernels):
            self.attentions.append(
                AttentionBlock2d(channels_x=f, channels_g=prev_channel, 
                                attention_channels=att_channels, kernel_size=3)
            )
            self.decoder.append(ResBlock(prev_channel + f, f, k))
            prev_channel = f
        self.final_resblock = ResBlock(prev_channel, filters[0], kernel_size[0])
        self.output_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)
    def forward(self, image, frequencies, freq_mask):
        if self.fusion_type != 'none':
            freq_features, freq_seq = self.freq_encoder(frequencies, freq_mask)
        else:
            freq_features, freq_seq = None, None
        if self.fusion_type == 'early':
            image = self.early_fusion(image, freq_features)
        x = self.conv(image)
        skips = []
        for resblock in self.encoder:
            x = resblock(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        x = self.bottleneck_dropout(x)
        if self.fusion_type == 'late':
            x = self.late_fusion(x, freq_features)
        for i, (resblock, attn_block) in enumerate(zip(self.decoder, self.attentions)):
            skip = skips.pop()
            if self.fusion_type in ['attention', 'hybrid']:
                # freq_mask may have a different length than freq_seq's token dim
                # (e.g. SpectralCNNEncoder pools 100 bins → 25 tokens). In that
                # case freq_mask cannot be used as key_padding_mask; pass None
                # instead (all tokens are real — no padding in binned encoding).
                cross_mask = (
                    freq_mask
                    if (freq_mask is None or freq_seq is None
                        or freq_mask.shape[-1] == freq_seq.shape[-1])
                    else None
                )
                skip, _ = self.cross_attentions[i](skip, freq_seq, cross_mask)
            attn_skip, _ = attn_block(x, skip)
            if self.fusion_type in ['film', 'hybrid']:
                attn_skip = self.film_layers[i](attn_skip, freq_features)
            x = self.up(x)
            x = torch.cat((x, attn_skip), dim=1)
            x = resblock(x)
        x = self.final_resblock(x)
        x = self.output_conv(x)
        return x
