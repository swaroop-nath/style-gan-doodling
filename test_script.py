from generator import Generator
from discriminator import Discriminator
from numpy import log2
import torch

if __name__ == "__main__":
    Z_DIM = 100
    # IN_CHANNELS = 256
    gen = Generator(32, Z_DIM, img_channels=3)
    critic = Discriminator(32, img_channels=3)

    for img_size in [4, 8, 16, 32]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((Z_DIM))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}\n\n")