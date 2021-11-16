from typing import OrderedDict
import torch
import torch.nn as nn
from blocks import EqualizedConvBlock, InitialGBlock, SynthesisBlock
import numpy as np
from custom_layers import EqualizedConv2dLayer, EqualizedLinearLayer

# class MappingNetwork(nn.Module):
#     def __init__(self, latent_dim, num_layers=8):
#         super().__init__()
#         layers = []
#         for i in range(num_layers):
#             layer = EqualizedLinearLayer(latent_dim, latent_dim)
#             layers.append(('mapping_dense_{}'.format(i), layer))

#         self.map_net = nn.Sequential(OrderedDict(layers))

#     def forward(self, z):
#         w = self.map_net(z)
#         return w

class MappingNetwork(nn.Module):
    def __init__(self, in_size, in_ch, latent_dim):
        super().__init__()
        assert in_size <= 64, 'IMG_SIZE is atmost 64 x 64'

        self.out_size = 4
        possible_channels = [16, 32, 64, 128, 256]
        num_enc_blocks = int(np.log2(in_size/self.out_size)) + 1
        self.channels = possible_channels[:num_enc_blocks]

        self.keyword_args = self._form_keyword_args()

        enc_blocks = []
        for idx in range(len(self.channels)):
            out_ch = self.channels[idx]
            block = EqualizedConvBlock(in_ch=in_ch, out_ch=out_ch, kernel_size=self.keyword_args['kernel-size'], **self.keyword_args)
            in_ch = out_ch
            enc_blocks.append(block)

        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.dense = EqualizedLinearLayer(in_size=self.channels[-1] * (self.out_size ** 2), out_size=latent_dim, **self.keyword_args)

    def forward(self, img):
        cond_feat_maps = []
        for idx, block in enumerate(self.enc_blocks):
            img = block(img)
            if idx != 0: img = self.downsample(img, kernel_size=self.keyword_args['downsample-kernel-size'], stride=self.keyword_args['downsample-stride'])
            cond_feat_maps.append(img)

        img = img.view(-1, self.channels[-1] * self.out_size ** 2) # (batch_size, channels, 4, 4) --> (batch_size, channels*16)
        latent_vector = self.dense(img)

        return latent_vector, cond_feat_maps

    def downsample(self, img, kernel_size, stride):
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)(img)

    def _form_keyword_args(self):
        return {'_padding': 1, 'p-mode': 'zeros', '_stride': 1, 'gain': 2**0.5, 
        'l-relu-slope': 0.2, '_activation': 'l-relu', 'use-pn': True, 
        'epsilon': 1e-8, 'kernel-size': 3, 'downsample-kernel-size': 2, 'downsample-stride': 2}
        

class Generator(nn.Module):
    def __init__(self, out_size, latent_dim, enc_channels, img_channels=3, concat_map=True):
        super().__init__()
        assert out_size <= 64, 'IMG_SIZE is atmost 64 x 64'

        self.start_size = 4
        possible_channels = [512, 256, 128, 64, 32] # Corresponding to img size == 4, 8, 16, 32, . . .
        num_synthesis_blocks = int(np.log2(out_size/self.start_size)) + 1
        self.channels = possible_channels[-num_synthesis_blocks:]
        if concat_map: self.enlarged_channels = [channel_enc + channel_gen for channel_enc, channel_gen in zip(enc_channels, self.channels)]
        else: self.enlarged_channels = self.channels

        self.concat_map = concat_map

        keyword_args = self._form_keyword_args()
        temp_kw_args = {'gain': 2**0.5}

        # TODO: Update each conv block to accomodate for concatenation from mapping network

        self.initial = InitialGBlock(self.start_size, self.enlarged_channels[0], self.channels[0], latent_dim, **keyword_args)
        self.initial_rgb = EqualizedConv2dLayer(in_ch=self.enlarged_channels[0], out_ch=img_channels, kernel_size=1, padding=0, padding_mode=None, stride=1, **temp_kw_args)

        synthesis_blocks = []
        self.rgb_layers = []
        for idx in range(num_synthesis_blocks - 1):
            block = SynthesisBlock(in_ch=self.enlarged_channels[idx], out_ch=self.channels[idx+1], latent_dim=latent_dim, upsample_mode='bilinear', **keyword_args)
            synthesis_blocks.append(block)
            self.rgb_layers.append(EqualizedConv2dLayer(in_ch=self.enlarged_channels[idx+1], out_ch=img_channels, kernel_size=1, padding=0, padding_mode=None, stride=1, **temp_kw_args))

        self.synthesis_net = nn.ModuleList(synthesis_blocks)

    def forward(self, latent_vector, alpha, steps, cond_feature_maps):

        assert steps <= len(self.synthesis_net), 'Number of steps can\'t be more than the number of blocks available'

        image = self.initial(latent_vector)
        if self.concat_map: image = torch.cat([image, cond_feature_maps[-1]], dim=1) # img_shape = (batch_size, channels, width, height)
        if steps == 0: return self.initial_rgb(image)

        curr_rgb = self.initial_rgb(image)
        prev_rgb = None
        for idx in range(steps):
            image = self.synthesis_net[idx](image, latent_vector)
            if self.concat_map: image = torch.cat([image, cond_feature_maps[-(idx+2)]], dim=1)
            prev_rgb = curr_rgb
            curr_rgb = self.rgb_layers[idx](image)

        prev_rgb_upsampled = self.upsample(prev_rgb, upsample_mode='bilinear')

        return self.fade_in(alpha, curr_rgb, prev_rgb_upsampled)
        

    def upsample(self, image, upsample_mode):
        return nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)(image)

    def fade_in(self, alpha, fading_in_img, existent_block_img):
        return alpha * fading_in_img + (1 - alpha) * existent_block_img

    def _form_keyword_args(self):
        return {'_padding': 1, 'p-mode': 'zeros', '_stride': 1, 'gain': 2**0.5, 
        'l-relu-slope': 0.2, '_activation': 'l-relu', 'use-pn': True, 
        'epsilon': 1e-8, 'kernel-size': 3}
        