# -*- coding: utf-8 -*-
# Author: Xingcheng Xu
# Rewrite: Yongkang Lin
"""
# BSDE-Gen Model
"""

import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.blocks import ResBlock, UnetResBlock
from typing import Optional, Sequence, Tuple, Union


class Decoder(nn.Module):
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[Sequence[int], int],
                 stride: Union[Sequence[int], int],
                 norm_name: Union[Tuple, str] = "instance"
                 ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.conv_block = UnetResBlock(
            spatial_dims,
            in_channels + in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name,
        )

    def forward(self, inp, skip):
        out = torch.cat((inp, skip), dim=1)
        out = self.conv_block(out)
        out = self.upsample(out)
        return out


class Z_nn(nn.Module):
    def __init__(self, norm_name="instance", num_of_res=6):
        """
        Args:
            norm_name:归一化方法
            num_of_res: 残差块的数量
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad3d((3, 3, 3, 3, 3, 3)),
            nn.Conv3d(3, 8, kernel_size=7, stride=1),
            nn.InstanceNorm3d(8),
            nn.ReLU()
        )
        # self.pad1 = nn.ReflectionPad3d((3, 3, 3, 3, 3, 3))
        # self.conv1 = nn.Conv3d(1, 8, kernel_size=7, stride=1)
        # self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        # self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate')

        self.conv2 = UnetResBlock(3, 8, 16, kernel_size=3, stride=2, norm_name=norm_name)
        self.conv3 = UnetResBlock(3, 16, 32, kernel_size=3, stride=2, norm_name=norm_name)
        self.conv4 = UnetResBlock(3, 32, 64, kernel_size=3, stride=2, norm_name=norm_name)
        self.conv5 = UnetResBlock(3, 64, 128, kernel_size=3, stride=2, norm_name=norm_name)

        self.r_blocks = nn.Sequential(
            *[ResBlock(3, 128, kernel_size=3, norm=norm_name) for _ in range(num_of_res)]
        )
        # self.deconv3 = nn.Conv3d(64, 16, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        # self.deconv2 = nn.Conv3d(32, 8, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        # self.deconv1 = nn.Conv3d(16, 1, kernel_size=7, stride=1)

        self.deconv5 = Decoder(3, 128, 64, 3, 1, norm_name=norm_name)
        self.deconv4 = Decoder(3, 64, 32, 3, 1, norm_name=norm_name)
        self.deconv3 = Decoder(3, 32, 16, 3, 1, norm_name=norm_name)
        self.deconv2 = Decoder(3, 16, 8, 3, 1, norm_name=norm_name)

        self.deconv1 = nn.Sequential(
            nn.ReflectionPad3d((3, 3, 3, 3, 3, 3)),
            nn.Conv3d(16, 1, kernel_size=7, stride=1),
            nn.InstanceNorm3d(1),
            nn.ReLU()
        )

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        res = self.r_blocks(conv5)

        de5 = self.deconv5(res, conv5)
        de4 = self.deconv4(de5, conv4)
        de3 = self.deconv3(de4, conv3)
        de2 = self.deconv2(de3, conv2)
        z = self.deconv1(torch.cat((de2, conv1), 1))
        return z


class FBSDEGen(nn.Module):
    def __init__(self, dim_x, dim_y, dim_h1=1000, dim_h2=600, dim_h3=1000, T=1.0, N=2,
                 device=None):
        super(FBSDEGen, self).__init__()

        self.T = T
        self.N = N
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_w = dim_x
        self.device = device

        self.z_nn = SwinUNETR(img_size=(128, 128, 32), in_channels=3, out_channels=1, use_v2=True)
        self.y0_nn = SwinUNETR(img_size=(128, 128, 32), in_channels=3, out_channels=1, use_v2=True)

        # self.channel_merge = nn.Conv3d(3, 1, kernel_size=1)

    def f(self, y, z):
        # z_abs = torch.abs(z)
        # z_term = 0.5 * (z_abs ** 2)
        # return z_term
        return torch.abs(z)

    def forward(self, input):
        # x = self.channel_merge(input)
        y = self.y0_nn(input)
        z = self.z_nn(input)

        for i in range(self.N):
            dw = torch.randn(z.size()).to(self.device) * 0.01
            y = (y - self.f(y, z) * 0.5 +
                 torch.matmul(z.permute(0, 1, 4, 2, 3), dw.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2))
        # y = y - self.f(y, z) * ( 1 / self.N )
        # y = y - self.f(y, z)
        return y


if __name__ == '__main__':
    cuda_id = 1
    device = torch.device(f'cuda:{cuda_id}')
    print(f"device={device}")

    dim_h1, dim_h2, dim_h3 = 64, 32, 64  # 1000, 600, 1000

    T, N = 1.0, 2
    dim_x = 224
    dim_y = 224
    dim_z = 16

    model = FBSDEGen(dim_x, dim_y, dim_h1, dim_h2, dim_h3, T, N, device)
    model = model.to(device)

    n_params = sum([p.numel() for p in model.parameters()])
    print(f"number of parameters: {n_params}")

    x = torch.randn(1, 3, 256, 256, 32).to(device)

    y = model(x)

    print(y.shape)
