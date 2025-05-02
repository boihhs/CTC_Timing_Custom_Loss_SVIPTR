"""
This was written by Surya.
I'm (Leo Benaharon) is uploading it because I'm not sure where he uploaded it and I want the other 
team members to access it
"""

import torch
import torch.nn as nn


# Defining Model
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)

    # Assume input is in (B x H x W x C)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # (B x C x H x W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = x.permute(0, 2, 3, 1)  # (B x H x W x C)
        return x