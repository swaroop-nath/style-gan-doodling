from generator import Generator, MappingNetwork
from discriminator import Discriminator
from numpy import log2
import torch

if __name__ == "__main__":
    Z_DIM = 100
    # IN_CHANNELS = 256
    map_net = MappingNetwork(in_size=32, in_ch=3, latent_dim=Z_DIM)
    gen = Generator(32, Z_DIM, list(reversed(map_net.channels)), img_channels=3, concat_map=True)
    critic = Discriminator(32, img_channels=3)

    for img_size in [4, 8, 16, 32]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, 3, 32, 32))
        w, feat_maps = map_net(x)
        z = gen(w, 0.5, steps=num_steps, cond_feature_maps=feat_maps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}\n\n")