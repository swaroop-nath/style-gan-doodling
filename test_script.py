from generator import Generator, MappingNetwork
from discriminator import Discriminator
from numpy import log2
import torch

if __name__ == "__main__":
    Z_DIM = 100
    # IN_CHANNELS = 256
    map_net = MappingNetwork(in_size=32, in_ch=10, latent_dim=Z_DIM)
    gen = Generator(32, Z_DIM, list(reversed(map_net.channels)), img_channels=1, concat_map=True)
    critic = Discriminator(32, img_channels=1)

    for img_size in [4, 8, 16, 32]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((4, 10, 32, 32))
        w, feat_maps = map_net(x)
        print('Map net asserted')
        z = gen(w, 0.5, steps=num_steps, cond_feature_maps=feat_maps)
        assert z.shape == (4, 1, img_size, img_size)
        print('gen asserted')
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (4, 1)
        print(f"Success! At img size: {img_size}\n\n")