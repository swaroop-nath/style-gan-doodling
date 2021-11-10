from typing import OrderedDict
import torch.nn as nn
from torch.nn.modules import padding
from blocks import EqualizedConvBlock, InitialGBlock, SynthesisBlock
import numpy as np
from custom_layers import EqualizedLinearLayer

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, num_layers=8):
        layers = []
        for i in range(num_layers):
            layer = EqualizedLinearLayer(latent_dim, latent_dim)
            layers.append(('mapping_dense_{}'.format(i), layer))

        self.map_net = nn.Sequential(OrderedDict(layers))

    def forward(self, z):
        w = self.map_net(z)
        return w

class Generator(nn.Module):
    def __init__(self, out_size, latent_dim, img_channels=3):
        assert out_size <= 64, 'IMG_SIZE is atmost 64 x 64'

        self.start_size = 4
        self.out_size = out_size
        possible_channels = [512, 256, 128, 64, 32] # Corresponding to img size == 4, 8, 16, 32, . . .
        num_synthesis_blocks = int(np.log2(out_size/self.start_size)) + 1
        self.channels = possible_channels[-num_synthesis_blocks:]

        self.latent_dim = latent_dim

        keyword_args = self._form_keyword_args()

        self.initial = InitialGBlock(self.start_size, self.channels[0], self.channels[0], self.latent_dim, **keyword_args)

        synthesis_blocks = []
        self.rgb_layers = []
        for idx in range(num_synthesis_blocks - 1):
            # - 2 because one is an initial block and idx + 2
            block = SynthesisBlock(in_ch=self.channels[idx+1], out_ch=self.channels[idx+2], latent_dim=self.latent_dim, upsample_mode='bilinear' **keyword_args)
            synthesis_blocks.append(block)
            temp_kw_args = {'padding': 0, 'p-mode': None, 'stride': 1, 'gain': 2**0.5}
            self.rgb_layers.append(EqualizedConvBlock(in_ch=self.channels[idx+1], out_ch=img_channels, kernel_size=1, activation='l-relu', **temp_kw_args))

        self.synthesis_net = nn.ModuleList(synthesis_blocks)

    def fade_in(self, alpha, upscaled_img, old_net_img):
        return alpha * upscaled_img + (1 - alpha) * old_net_img

    
    def _form_keyword_args(self):
        pass
        