#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：YJXLK 
@File    ：HLFF.py
@IDE     ：PyCharm 
@Author  ：Linkang Xu
@Date    ：2025/2/24 13:32 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv2d, self).__init__()

        # Depthwise convolution (each input channel has its own filter)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)

        # Pointwise convolution (1x1 convolution to combine features across channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)  # Apply depthwise convolution
        x = self.pointwise(x)  # Apply pointwise convolution
        return x


class HighLowFeatureFusionModule(nn.Module):
    def __init__(self, C1, C2):
        super(HighLowFeatureFusionModule, self).__init__()

        # Depthwise Separable Convolution for 3x3 kernel
        self.depthwise_separable = DepthwiseSeparableConv2d(C1 + C2, C1, kernel_size=3, padding=1)

        # Depthwise Separable Convolution for 1x1 kernel (for dimensionality adjustment)


    def forward(self, low_res_features, high_res_features):
        # low_res_features: B, C2, H2, W2
        # high_res_features: B, C1, H1, W1
        B, C2, H2, W2 = low_res_features.size()
        B, C1, H1, W1 = high_res_features.size()

        # Upsample low_res_features (B, C2, H2, W2) to match the size of high_res_features (B, C2, H1, W1)
        low_res_upsampled = F.interpolate(low_res_features, size=(H1, W1), mode='bilinear', align_corners=False)

        # Concatenate the upsampled low-res features and high-res features along the channel dimension
        fused_features = torch.cat((high_res_features, low_res_upsampled), dim=1)  # Shape: B, C1+C2, H1, W1

        # Apply depthwise separable 3x3 convolution to fuse high and low features
        out = self.depthwise_separable(fused_features)

        return out

if __name__ == '__main__':

    # Example usage:
    # Assuming low_res_features with shape (B, C2, H2, W2) and high_res_features with shape (B, C1, H1, W1)
    low_res_features = torch.randn(4, 64, 32, 32)  # Example low-res feature map: B=4, C2=64, H2=32, W2=32
    high_res_features = torch.randn(4, 128, 64, 64)  # Example high-res feature map: B=4, C1=128, H1=64, W1=64

    fusion_module = HighLowFeatureFusionModule(C1=128, C2=64)
    output = fusion_module(low_res_features, high_res_features)
    print(output.shape)  # Should print torch.Size([4, 128, 64, 64]) since it's the same shape as high_res_features
