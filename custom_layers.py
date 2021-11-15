import torch.nn.functional as F
import torch.nn as nn
import torch

class EqualizedConv2dLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, padding_mode='zeros', stride=1, **kwargs):
        super().__init__()
        factor = in_ch * (kernel_size ** 2)
        self.he_scaler = he_init_scale(factor, kwargs['gain'])

        if padding_mode is not None: self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, stride=stride)
        else: self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv.bias = None
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        self.conv(x/self.he_scaler) + self.bias.view(1, self.bias.shape[0], 1, 1)

class NoiseLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_channels)) # weights are different for each channel

    def forward(self, img):
        noise = torch.randn(img.size(0), 1, img.size(2), img.size(3)) # noise across channels is constant
        img += self.weights.view(1, -1, 1, 1) * noise
        return img

class StyleMod(nn.Module):
    def __init__(self, latent_dim, channel_size, **kwargs):
        super().__init__()
        self.affine_stds = EqualizedLinearLayer(latent_dim, channel_size ** 2, **kwargs)
        self.affine_biases = EqualizedLinearLayer(latent_dim, channel_size ** 2, **kwargs)

    def forward(self, img, latent_vector):
        style_mod_stds = self.affine_stds(latent_vector).view(1, 1, img.size[-2], img.size[-1])
        style_mod_biases = self.affine_biases(latent_vector).view(1, 1, img.size[-2], img.size[-1])
        return style_mod_stds * img + style_mod_biases

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


