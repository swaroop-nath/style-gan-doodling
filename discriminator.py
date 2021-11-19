import torch
import torch.nn as nn
import numpy as np
from blocks import EqualizedConvBlock, FinalDBlock
from custom_layers import EqualizedConv2dLayer

class Discriminator(nn.Module):
    def __init__(self, in_size, img_channels=3):
        super().__init__()
        assert in_size <= 64, 'IMG_SIZE is atmost 64 x 64'

        self.start_size = 4
        possible_channels = [32, 64, 128, 256, 512]
        num_disc_blocks = int(np.log2(in_size/self.start_size)) + 1
        self.channels = possible_channels[:num_disc_blocks]

        self.keyword_args = self._form_keyword_args()
        temp_kw_args = {'gain': 2**0.5}

        disc_blocks = []
        self.rgb_layers = []
        for idx in range(num_disc_blocks - 1):
            self.rgb_layers.append(EqualizedConv2dLayer(in_ch=img_channels, out_ch=self.channels[idx], kernel_size=1, padding=0, padding_mode=None, stride=1, **temp_kw_args))
            disc_block = EqualizedConvBlock(in_ch=self.channels[idx], out_ch=self.channels[idx+1], kernel_size=self.keyword_args['kernel-size'], **self.keyword_args)
            disc_blocks.append(disc_block)

        self.disc_net = nn.ModuleList(disc_blocks)

        final_rgb = EqualizedConv2dLayer(in_ch=img_channels, out_ch=self.channels[-1], kernel_size=1, padding=0, padding_mode=None, stride=1, **temp_kw_args)
        self.rgb_layers.append(final_rgb)
        self.final = FinalDBlock(self.start_size, self.channels[-1], self.channels[-1], **self.keyword_args)

    def forward(self, img, alpha, steps):
        # 1. steps == 0: means only the final block, 1 means 8 x 8 -> 4 x 4 and so on
        if steps == 0:
            img = self.rgb_layers[-1](img)
            img = self.minibatch_std(img)
            return self.final(img)

        high_res_img = self.rgb_layers[-(steps+1)](img)
        high_res_img = self.disc_net[-steps](high_res_img)
        low_res_img_from_new_block = self.downsample(high_res_img, self.keyword_args['downsample-kernel-size'], self.keyword_args['downsample-stride'])

        downsampled_img = self.downsample(img, self.keyword_args['downsample-kernel-size'], self.keyword_args['downsample-stride'])
        low_res_img = self.rgb_layers[-steps](downsampled_img)

        img = self.fade_in(alpha, low_res_img_from_new_block, low_res_img)

        for idx in range(-steps+1, 0):
            img = self.disc_net[idx](img)
            img = self.downsample(img, self.keyword_args['downsample-kernel-size'], self.keyword_args['downsample-stride'])

        img = self.minibatch_std(img)
        return self.final(img)

    def fade_in(self, alpha, fading_in_img, existent_block_img):
        return alpha * fading_in_img + (1 - alpha) * existent_block_img

    def downsample(self, img, kernel_size, stride):
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)(img)

    def minibatch_std(self, x):
        # Reference - https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
        def find_batch_stat(x, group_size=5, num_new_features=1):
            b, c, h, w = x.shape
            assert b % group_size == 0, 'MiniBatch STDDEV: group size must divide batch size'

            group_size = min(group_size, b)
            y = x.reshape([group_size, -1, num_new_features,
                        c // num_new_features, h, w])
            y = y - y.mean(0, keepdim=True)
            y = (y ** 2).mean(0, keepdim=True)
            y = (y + 1e-8) ** 0.5
            y = y.mean([3, 4, 5], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
            y = y.expand(group_size, -1, -1, h, w).clone().reshape(b, num_new_features, h, w)
            return y

        # batch_stat = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        batch_stat = find_batch_stat(x)
        return torch.cat([x, batch_stat], dim=1)

    def _form_keyword_args(self):
        return {'_padding': 1, 'p-mode': 'zeros', '_stride': 1, 'gain': 2**0.5, 
        'l-relu-slope': 0.2, '_activation': 'l-relu', 'use-pn': True, 
        'epsilon': 1e-8, 'kernel-size': 3, 'end-kernel-size': 4, 'end-padding': 0,
        'downsample-kernel-size': 2, 'downsample-stride': 2, 'intermediate-units': 32,
        'use-one-conv-layer': True}