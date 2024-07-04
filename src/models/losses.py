from copy import deepcopy
import torch
from torch import nn
import importlib

from torch.nn import BCELoss, MSELoss  # Enable by default


class RMSLELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(*args, **kwargs)

    def forward(self, pred, actual):
        return self.mse(
            torch.log(torch.clamp(pred, 0) + 1), torch.log(torch.clamp(actual, 0) + 1)
        )


class HSGeneratorLoss(nn.Module):
    def __init__(
        self,
        gan_loss,
        radius_loss,
        grid_density_loss,
    ):
        super().__init__()
        self.gan_loss = gan_loss
        self.gan_loss_fn = BCELoss()
        self.radius_loss = radius_loss
        self.grid_density_loss = grid_density_loss

        # For logging, save the results
        self.prev_gan_loss = torch.tensor([0])
        self.prev_radius_loss = torch.tensor([0])
        self.prev_grid_density_loss = torch.tensor([0])

        self.mse = MSELoss()

    @staticmethod
    def make_subgrid_mask(lims_x, lims_y, n_x, n_y):
        lox = lims_x[0]
        hix = lims_x[1]

        loy = lims_y[0]
        hiy = lims_y[1]

        # Break x-y in evenly distributed grid squares

        grid_x = torch.linspace(lox, hix, steps=n_x - 1)

        grid_y = torch.linspace(loy, hiy, steps=n_y - 1)

    def _radius_loss(self, real_images, fake_images):
        # Loss based on sum of radiuses
        real_radius = real_images[:, 2].mean()
        fake_radius = fake_images[:, 2].mean()
        radius_loss = torch.abs(real_radius - fake_radius)

        # Loss based on the variance of radii
        real_r_var = real_images[:, 2].var()
        fake_r_var = fake_images[:, 2].var()
        r_var_loss = torch.abs(real_r_var - fake_r_var)

        # TODO Add more physical properties, phi etc.
        loss = (radius_loss + r_var_loss) / 2
        return loss

    def _gan_loss(self, fake_outputs, real_labels):
        return self.gan_loss_fn(fake_outputs, real_labels)

    @staticmethod
    def _non_saturating_gan_loss(fake_outputs):
        return -torch.mean(torch.log(fake_outputs))

    def _grid_density_loss(self, real_images, fake_images):

        quantiles = torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.95],
            dtype=torch.float32,
            device=fake_images.device,
        )
        # first two columns are x and y coordinates
        # Count the 5, 25, 50, 75, 95 quantiles for the first two columns and compare to the ground truth

        fake_xq = torch.quantile(fake_images[:, 0], quantiles)
        real_xq = torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.95], device=fake_images.device
        )  # NOTE: Tensors normalized from 0 to 1 so the quantiles should match

        fake_yq = torch.quantile(fake_images[:, 1], quantiles)
        real_yq = torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.95], device=fake_images.device
        )  # NOTE: Tensors normalized from 0 to 1 so the quantiles should match

        loss = (self.mse(fake_xq, real_xq) + self.mse(fake_yq, real_yq)) / 2

        return loss

    def forward(self, real_images, fake_images, fake_outputs, real_labels):

        loss = 0

        if self.radius_loss:
            self.prev_radius_loss = self._radius_loss(
                real_images=real_images, fake_images=fake_images
            )
            loss += self.prev_radius_loss

        if self.gan_loss:
            # self.prev_gan_loss = self._gan_loss(fake_outputs, real_labels)
            self.prev_gan_loss = self._non_saturating_gan_loss(fake_outputs)
            loss += self.prev_gan_loss

        if self.grid_density_loss:
            self.prev_grid_density_loss = self._grid_density_loss(
                real_images=real_images, fake_images=fake_images
            )
            loss += self.prev_grid_density_loss

        return loss


def build_loss_fn(**loss_config):

    name = loss_config.pop("name")

    module = importlib.import_module(".", package="src.models.losses")
    loss_fn_builder = getattr(module, name)

    loss_fn = loss_fn_builder(**loss_config)

    return loss_fn
