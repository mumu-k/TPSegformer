#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：YJXLK
@File    ：HBTrans.py
@IDE     ：PyCharm
@Author  ：Linkang Xu
@Date    ：2025/2/3 20:27
'''
import torch
from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np


class HilbertTransformer:
    def __init__(self, order=None, H=None, W=None):
        """
        Initialize the Hilbert Transformer with a given order and dimensions.
        :param order: The order of the Hilbert curve (typically log2 of the image size).
        :param H: Height of the image (used for expanding input).
        :param W: Width of the image (used for expanding input).
        """
        self.order = order
        self.H = H
        self.W = W

    def hilbert_curve_index(self, x):
        """
        Flatten input using a Hilbert curve indexing (sorted by Hilbert curve distance).
        :param x: Input tensor of shape (B, C, H, W)
        :return: Flattened tensor with Hilbert curve indexing applied.
        """
        B, C, H, W = x.shape
        hilbert_order = int(torch.log2(torch.tensor(H, dtype=torch.float32)))  # Hilbert curve order
        hilbert_curve = HilbertCurve(hilbert_order, 2)  # Initialize Hilbert curve

        # Generate Hilbert curve indices for 2D coordinates
        indices = []
        for i in range(H):
            for j in range(W):
                indices.append(hilbert_curve.distance_from_coordinates([i, j]))

        # Convert indices to a tensor and sort according to Hilbert curve order
        indices = torch.tensor(indices)
        sorted_indices = torch.argsort(indices)  # Sort the indices for Hilbert curve ordering

        # Flatten the tensor and rearrange based on the sorted indices
        flattened = x.view(B, C, -1)  # Flatten the spatial dimensions
        sorted_tensor = flattened[:, :, sorted_indices]  # Rearranged according to Hilbert curve indices
        return sorted_tensor

    def inverse_hilbert_curve_index(self, x):
        """
        Expand input from a Hilbert curve indexing (rearranged back to original spatial dimensions).
        :param x: Input tensor of shape (B, C, N), where N is H * W
        :return: Expanded tensor with Hilbert curve indexing restored to (B, C, H, W)
        """
        B, C, N = x.shape
        H = W = int(N ** 0.5)  # H and W are the original spatial dimensions

        hilbert_order = int(torch.log2(torch.tensor(H, dtype=torch.float32)))  # Hilbert curve order
        hilbert_curve = HilbertCurve(hilbert_order, 2)  # Initialize Hilbert curve

        # Generate Hilbert curve indices for 2D coordinates
        indices = []
        for i in range(H):
            for j in range(W):
                indices.append(hilbert_curve.distance_from_coordinates([i, j]))

        indices = torch.tensor(indices)
        sorted_indices = torch.argsort(indices)  # Sort the indices for Hilbert curve ordering
        inverse_indices = torch.argsort(sorted_indices)  # Get the inverse sorted indices

        # Rearrange the tensor based on the inverse indices and reshape
        expanded = x[:, :, inverse_indices].view(B, C, H, W)
        return expanded


if __name__ == '__main__':

    # 示例使用
    B, C, H, W = 1, 1, 4, 4  # 示例尺寸
    input_tensor = torch.randn(B, C, H, W)  # 随机输入张量
    print(input_tensor)
    # 初始化 HilbertTransformer
    hilbert_transformer = HilbertTransformer(order=int(torch.log2(torch.tensor(H, dtype=torch.float32))), H=H, W=W)

    # 降维操作
    reduced_tensor = hilbert_transformer.hilbert_curve_index(input_tensor)
    print("Reduced tensor shape:", reduced_tensor.shape)
    print(reduced_tensor)
    # 升维操作
    expanded_tensor = hilbert_transformer.inverse_hilbert_curve_index(reduced_tensor)
    print("Expanded tensor shape:", expanded_tensor.shape)
    print(expanded_tensor)










