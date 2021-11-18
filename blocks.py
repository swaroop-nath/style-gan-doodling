import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers import EqualizedConv2dLayer, EqualizedLinearLayer, NoiseLayer, StyleMod

class InitialGBlock(nn.Module):
    def __init__(self, start_img_size, start_channel, out_ch, latent_dim, **kwargs):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, start_channel, start_img_size, start_img_size)) # const IMG shape - (1, channels, width, height)
        self.noise1 = NoiseLayer(start_channel)
        self.style_transfer1 = StyleMod(latent_dim, start_channel, **kwargs)
        self.conv = EqualizedConvBlock(start_channel, out_ch, kwargs['kernel-size'], **kwargs)
        self.noise2 = NoiseLayer(out_ch)
        self.style_transfer2 = StyleMod(latent_dim, out_ch, **kwargs)

    def forward(self, latent_vector):
        img = self.noise1(self.const)
        
        # Instance normalization
        img = self.instance_norm(img)

        img = self.style_transfer1(img, latent_vector)

        img = self.conv(img)
        img = self.noise2(img)

        # Instance normalization
        img = self.instance_norm(img)
        img = self.style_transfer2(img, latent_vector)

        return img

    def instance_norm(self, img):
        # IMG shape - (batch, channels, width, height)
        mean = torch.mean(img, dim=(2, 3), keepdim=True) # Computing mean for each feature map
        std = torch.std(img, dim=(2, 3), keepdim=True) # Computing std dev for each feature map

        return (img - mean)/std

class FinalDBlock(nn.Module):
    def __init__(self, end_img_size, end_channel, out_ch, **kwargs):
        super().__init__()
        self.conv1 = EqualizedConv2dLayer(in_ch=end_channel+1, out_ch=out_ch, kernel_size=kwargs['kernel-size'], padding=kwargs['_padding'], padding_mode=kwargs['p-mode'], stride=kwargs['_stride'], **kwargs)
        self.conv2 = EqualizedConv2dLayer(in_ch=out_ch, out_ch=out_ch, kernel_size=kwargs['end-kernel-size'], padding=kwargs['end-padding'], padding_mode=kwargs['p-mode'], stride=kwargs['_stride'], **kwargs)

        activation = kwargs['_activation']
        if activation == 'l-relu':
            self.act1 = nn.LeakyReLU(negative_slope=kwargs['l-relu-slope'])
            self.act2 = nn.LeakyReLU(negative_slope=kwargs['l-relu-slope'])
        
        if activation == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()

        self.dense = EqualizedLinearLayer(in_size=1 * 1 * out_ch, out_size=1, **kwargs) # Fake or not fake

    def forward(self, img):
        # Image is already concatenated with minibatch std
        img = self.act1(self.conv1(img))
        img = self.act2(self.conv2(img))

        img = torch.flatten(img, start_dim=1) # Flattening each img in the batch

        logits = self.dense(img) # WGAN works with logits, not probs
        return logits


class SynthesisBlock(nn.Module):
    # Things to do - upsample, apply conv2d, add noise, apply ADAIN, apply conv2d, add noise and apply ADAIN
    def __init__(self, in_ch, out_ch, latent_dim, upsample_mode='bilinear', **kwargs):
        super().__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
        self.upsample = nn.PixelShuffle(upscale_factor=2) # Reference - https://github.com/soumith/ganhacks#authors

        self.conv1 = EqualizedConvBlock(in_ch, out_ch, kwargs['kernel-size'], **kwargs)
        self.noise1 = NoiseLayer(num_channels=out_ch)
        self.style_transfer1 = StyleMod(latent_dim, out_ch, **kwargs)

        self.conv2 = EqualizedConvBlock(out_ch, out_ch, kwargs['kernel-size'], **kwargs)
        self.noise2 = NoiseLayer(num_channels=out_ch)
        self.style_transfer2 = StyleMod(latent_dim, out_ch, **kwargs)

    def forward(self, img, latent_vector):
        img = self.upsample(img)
        img = self.conv1(img)
        img = self.noise1(img)

        # Instance normalization
        img = self.instance_norm(img)

        img = self.style_transfer1(img, latent_vector)

        img = self.conv2(img)
        img = self.noise2(img)

        # Instance normalization
        img = self.instance_norm(img)

        img = self.style_transfer2(img, latent_vector)

        return img

    def instance_norm(self, img):
        mean = torch.mean(img, dim=[2, 3], keepdim=True)
        std = torch.std(img, dim=[2, 3], keepdim=True)

        return (img - mean)/std


class EqualizedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, **kwargs):
        super().__init__()
        self.conv1 = EqualizedConv2dLayer(in_ch, out_ch, kernel_size, padding=kwargs['_padding'], padding_mode=kwargs['p-mode'], stride=kwargs['_stride'], gain=kwargs['gain'])
        self.conv2 = EqualizedConv2dLayer(out_ch, out_ch, kernel_size, padding=kwargs['_padding'], padding_mode=kwargs['p-mode'], stride=kwargs['_stride'], gain=kwargs['gain'])
        activation = kwargs['_activation']
        if activation == 'l-relu':
            self.act1 = nn.LeakyReLU(negative_slope=kwargs['l-relu-slope'])
            self.act2 = nn.LeakyReLU(negative_slope=kwargs['l-relu-slope'])
        
        if activation == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()

        self.use_pixel_norm = kwargs['use-pn']
        self.epsilon = kwargs['epsilon']

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pixel_norm(x) if self.use_pixel_norm else x
        x = self.act2(self.conv2(x))
        x = self.pixel_norm(x) if self.use_pixel_norm else x

        return x

    def pixel_norm(self, img):
        return img / torch.sqrt(torch.mean(img ** 2, dim=1, keepdim=True) + self.epsilon)

