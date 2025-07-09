#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：YJXLK 
@File    ：LCCM.py
@IDE     ：PyCharm 
@Author  ：Linkang Xu
@Date    ：2025/2/24 13:48 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from hilbertcurve.hilbertcurve import HilbertCurve
from thop import profile
from hbtrans import HilbertTransformer
import time
import math




class LayerCorrelationCalculation(nn.Module):
    def __init__(self, input_size=32,in_channel=13,mode = "row-first"):
        super(LayerCorrelationCalculation, self).__init__()
        self.input_size = input_size
        self.in_channel = in_channel
        self.q_conv = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=1)  # Generate Q
        self.k_conv = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=1)  # Generate K
        self.v_conv = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=1)  # Generate V

        self.qkv = nn.Sequential(
            nn.Conv2d(self.in_channel,3,kernel_size=1),
        )

        self.mode = mode

        self.sigmoid = nn.Sigmoid()

    def hilbert_curve_index(self, x):
        """
        Flatten input using a Hilbert curve indexing.
        """
        B, C, H, W = x.shape
        hilbert_order = int(torch.log2(torch.tensor(H, dtype=torch.float32)))
        hilbert_transformer = HilbertTransformer(hilbert_order,H,W)
        return hilbert_transformer.hilbert_curve_index(x)


    def inverse_hilbert_curve_index(self, x):
    #     """
    #     Expand input from a Hilbert curve indexing.
    #     """
        B, C, N = x.shape
        H = W = int(N ** 0.5)
        hilbert_order = int(torch.log2(torch.tensor(H, dtype=torch.float32)))
        hilbert_transformer = HilbertTransformer(hilbert_order, H, W)
        return hilbert_transformer.inverse_hilbert_curve_index(x)

    def z_curve_index(self, x):
        """
        Flatten input using Z-curve indexing.
        """
        B, C, H, W = x.shape

        def z_order(i, j):
            value = 0
            bit = 0
            while (1 << bit) < max(H, W):
                value |= ((i & (1 << bit)) << bit) | ((j & (1 << bit)) << (bit + 1))
                bit += 1
            return value

        indices = []
        for i in range(H):
            for j in range(W):
                indices.append(z_order(i, j))

        sorted_indices = torch.argsort(torch.tensor(indices))
        flattened = x.view(B, C, -1)
        return flattened[:, :, sorted_indices]

    def inverse_z_curve_index(self, x):
        """
        Expand input from a Z-curve indexing.
        """
        B, C, N = x.shape
        H = W = int(N ** 0.5)

        def z_order(i, j):
            value = 0
            bit = 0
            while (1 << bit) < max(H, W):
                value |= ((i & (1 << bit)) << bit) | ((j & (1 << bit)) << (bit + 1))
                bit += 1
            return value

        indices = []
        for i in range(H):
            for j in range(W):
                indices.append(z_order(i, j))

        sorted_indices = torch.argsort(torch.tensor(indices))
        inverse_indices = torch.argsort(sorted_indices)
        expanded = x[:, :, inverse_indices].view(B, C, H, W)
        return expanded

    def flatten(self, x, mode="row-first"):
        """
        Flatten the input tensor according to the specified mode.
        """
        B, C, H, W = x.shape
        if mode == "row-first":
            return x.view(B, C, -1)
        elif mode == "hilbert":
            return self.hilbert_curve_index(x)
        elif mode == "z-curve":
            return self.z_curve_index(x)
        else:
            raise ValueError(f"Unsupported flattening mode: {mode}")

    def expand(self, x, mode="row-first"):
        """
        Expand the input tensor back to spatial dimensions according to the specified mode.
        """
        B, C, N = x.shape
        H = W = int(N ** 0.5)
        if mode == "row-first":
            return x.view(B, C, H, W)
        elif mode == "hilbert":
            return self.inverse_hilbert_curve_index(x)
        elif mode == "z-curve":
            return self.inverse_z_curve_index(x)
        else:
            raise ValueError(f"Unsupported expansion mode: {mode}")

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.input_size and W == self.input_size, "Input size mismatch"

        x_max,_ = torch.max(x,dim=1,keepdim=True)


        # Generate Q, K, V

        qkv = self.qkv(x_max)
        Q,K,V = torch.split(qkv,1,dim=1)

        # Flatten for operations
        Q_flat = self.flatten(Q, self.mode)  # [B, C, H*W]
        K_flat = self.flatten(K, self.mode)
        V_flat = self.flatten(V, self.mode)


        # Attention operation 2
        attention_map = torch.matmul(Q_flat.transpose(-1, -2), K_flat)  # [B, H*W, H*W]
        attention_map = F.softmax(attention_map, dim=-2)  # Apply softmax
        output = torch.matmul(V_flat,attention_map)  # [B, C, H*W]
        output = self.expand(output, self.mode)  # Reshape to [B, C, H, W]

        output = x*self.sigmoid(output)

        return output

if __name__ == '__main__':
    x = torch.randn(4, 256, 32, 32).cuda()
    model = LayerCorrelationCalculation(32,1,mode="hilbert").cuda()
    y = model(x)
    print(y.shape)