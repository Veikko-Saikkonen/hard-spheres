import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, channels_img=3, features_g=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features_g * 2, channels_img, 4, 2, 1),  # 64x64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, channels_img=3, features_d=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),  # 1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


# Hyperparameters
z_dim = 100
lr = 0.0002
batch_size = 128
num_epochs = 5
channels_img = 1
features_d = 64
features_g = 64


class HardSphereGAN:
    def __init__(
        self,
        z_dim=100,
        lr=0.0002,
        batch_size=128,
        image_size=64,
        channels_img=1,
        features_d=64,
        features_g=64,
    ):
        self.z_dim = z_dim
        self.lr = lr
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels_img = channels_img
        self.features_d = features_d
        self.features_g = features_g

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gen = Generator(z_dim, channels_img, features_g).to(self.device)
        self.disc = Discriminator(channels_img, features_d).to(self.device)

        self.opt_gen = torch.optim.Adam(
            self.gen.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.opt_disc = torch.optim.Adam(
            self.disc.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(64, z_dim, 1, 1).to(self.device)

    def _reset_gradients(self):
        self.opt_gen.zero_grad()
        self.opt_disc.zero_grad()

    def _discriminator_loss(self, real, fake):
        real_loss = self.criterion(real, torch.ones_like(real))
        fake_loss = self.criterion(fake, torch.zeros_like(fake))
        return real_loss + fake_loss

    def _generator_loss(self, fake):
        return self.criterion(fake, torch.ones_like(fake))

    def forward(self, x):
        return self.gen(x)

    # Create a basic training method for the GAN

    def train(self, real):
        self.gen.train()
        self.disc.train()

        real = real.to(self.device)
        noise = torch.randn(batch_size, z_dim, 1, 1).to(self.device)
        fake = self.gen(noise)
        disc_real = self.disc(real).reshape(-1)
        loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = self.disc(fake.detach()).reshape(-1)
        loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        self._reset_gradients()
        loss_disc.backward(retain_graph=True)
        self.opt_disc.step()

        output = self.disc(fake).reshape(-1)
        loss_gen = self.criterion(output, torch.ones_like(output))
        self._reset_gradients()
        loss_gen.backward()
        self.opt_gen.step()

        return loss_disc, loss_gen
