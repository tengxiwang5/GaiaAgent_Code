import torch
from torch import nn
from torch.nn import functional as F


class InformationInteraction(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(InformationInteraction, self).__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, feat_t1, feat_t2):
        B, C, H, W = feat_t1.size()

        query_t1 = self.query_proj(feat_t1).view(B, C, -1).permute(0, 2, 1)
        key_t2 = self.key_proj(feat_t2).view(B, C, -1)
        value_t2 = self.value_proj(feat_t2).view(B, C, -1).permute(0, 2, 1)

        attention_t1_t2 = torch.matmul(query_t1, key_t2) * self.scale
        attention_t1_t2 = F.softmax(attention_t1_t2, dim=-1)
        interacted_feat_t1 = torch.matmul(attention_t1_t2, value_t2).permute(0, 2, 1).view(B, C, H, W)
        interacted_feat_t1 = self.output_proj(interacted_feat_t1) + feat_t1

        query_t2 = self.query_proj(feat_t2).view(B, C, -1).permute(0, 2, 1)
        key_t1 = self.key_proj(feat_t1).view(B, C, -1)
        value_t1 = self.value_proj(feat_t1).view(B, C, -1).permute(0, 2, 1)

        attention_t2_t1 = torch.matmul(query_t2, key_t1) * self.scale
        attention_t2_t1 = F.softmax(attention_t2_t1, dim=-1)
        interacted_feat_t2 = torch.matmul(attention_t2_t1, value_t1).permute(0, 2, 1).view(B, C, H, W)
        interacted_feat_t2 = self.output_proj(interacted_feat_t2) + feat_t2

        return interacted_feat_t1, interacted_feat_t2
