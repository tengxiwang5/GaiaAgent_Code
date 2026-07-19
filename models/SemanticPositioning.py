import torch
from torch import nn


class AdaptivePositionalEncoding(nn.Module):
    def __init__(self, d_model=128, num_patches=4096):
        super().__init__()
        self.adaptive_pe = nn.Linear(d_model, d_model)
        self.base_pe = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

    def forward(self, x):
        pe = self.base_pe + self.adaptive_pe(x)
        return x + pe


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)


class SemanticPositioning(nn.Module):
    def __init__(self, d_model=160, num_patches=4096, output_dim=512):
        super().__init__()
        self.pos_embed_layer = AdaptivePositionalEncoding(d_model=d_model, num_patches=num_patches)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.reduce_fc = nn.Linear(d_model, output_dim)
        self.transformer = TransformerEncoder(embed_dim=d_model)

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        feature_map = feature_map.view(B, C, H * W).permute(0, 2, 1)
        feature_map = self.pos_embed_layer(feature_map)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, feature_map], dim=1)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.reduce_fc(x)
        return x
