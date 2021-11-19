from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EqualizedConv2dLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, padding_mode='zeros', stride=1, **kwargs):
        super().__init__()
        factor = in_ch * (kernel_size ** 2)
        self.he_scaler = he_init_scale(factor, kwargs['gain'])

        if padding_mode is not None: self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, stride=stride).to(device)
        else: self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding, stride=stride).to(device)
        self.conv.bias = None
        self.bias = nn.Parameter(torch.zeros(out_ch)).to(device)

    def forward(self, x):
        # x - img of shape (channels, width, height)
        return self.conv(x/self.he_scaler) + self.bias.view(1, self.bias.shape[0], 1, 1)

class NoiseLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_channels)).to(device) # weights are different for each channel

    def forward(self, img):
        noise = torch.randn(1, 1, img.size(2), img.size(3)).to(device) # noise across channels is constant
        return img + self.weights.view(1, -1, 1, 1) * noise

class StyleMod(nn.Module):
    def __init__(self, latent_dim, channels, **kwargs):
        super().__init__()
        self.affine = EqualizedLinearLayer(latent_dim, channels * 2, **kwargs)

    def forward(self, img, latent_vector):
        # IMG shape - (batch, channels, width, height)
        style = self.affine(latent_vector)
        shape = [-1, 2, img.size(1)] + (img.dim() - 2) * [1] # shape == (batch_size, 2, num_channels, img_width, img_height)
        style = style.view(shape)

        # Reference - https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
        return img * (style[:, 0] + 1.) + style[:, 1] # Component 1 acts as the modulation and component 2 acts as the bias

class EqualizedLinearLayer(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
        super().__init__()

        he_scaler = he_init_scale(in_size, kwargs['gain'])
        self.weights = nn.Parameter(torch.randn(out_size, in_size) * he_scaler)
        self.bias = nn.Parameter(torch.zeros(out_size))

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)

def he_init_scale(factor, gain=2**0.5):
    return gain * (factor ** (-0.5))