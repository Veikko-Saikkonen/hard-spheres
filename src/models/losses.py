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
        physical_feasibility_loss,
        collision_loss_coefficient,
    ):
        super().__init__()
        self.gan_loss = gan_loss
        self.gan_loss_fn = BCELoss()
        self.radius_loss = radius_loss
        self.grid_density_loss = grid_density_loss
        self.physical_feasibility_loss = physical_feasibility_loss
        self.collision_loss_coefficient = collision_loss_coefficient

        # For logging, save the results
        self.prev_gan_loss = torch.tensor([0])
        self.prev_radius_loss = torch.tensor([0])
        self.prev_grid_density_loss = torch.tensor([0])
        self.prev_physical_feasibility_loss = torch.tensor([0])

        self.mse = MSELoss()

    def _radius_loss(self, real_images, fake_images):
        # Loss based on sum of radiuses
        quantiles = torch.tensor(  # TODO: Make this a parameter
            [0.05, 0.25, 0.50, 0.75, 0.95],
            dtype=torch.float32,
            device=fake_images.device,
        )

        fake_rq = torch.quantile(fake_images[:, :, 2], quantiles)
        real_yq = torch.quantile(real_images[:, :, 2], quantiles)
        loss = self.mse(fake_rq, real_yq)

        return loss

    def _physical_feasibility_loss(self, fake_points, collision_loss_coefficient=1):

        # Rescale the coordinates and radii back to their original scales
        x = fake_points[:, :, 0]
        y = fake_points[:, :, 1]
        radii = fake_points[:, :, 2]

        # Combine rescaled values into one tensor
        rescaled_points = torch.stack((x, y, radii), dim=2)
        # Calculate the pairwise distance matrix
        # If the distance is less than the sum of the radii, then it's a collision

        # For each collision, add a penalty

        # Calculate the pairwise distance matrix
        n = fake_points.shape[1]
        dist = torch.cdist(rescaled_points[:, :, :2], rescaled_points[:, :, :2])
        # Calculate the sum of the radii of the two points
        # TODO: This is not correct as the points and radii are scaled with different measures
        # TODO: Think of this
        # NOTE: Will not work correctly as this will result in points being counted as overlapping even if they are not overlapping
        radii = (
            rescaled_points[:, :, 2].unsqueeze(1).abs()
            + rescaled_points[:, :, 2].unsqueeze(2).abs()
        )

        # Calculate the collision matrix
        # collision_matrix = radii > dist

        # Instead of a hard limit, use a differentiable loss function
        overlap_distance = (radii - dist) / radii.sum()
        overlap_distance = nn.ReLU(inplace=False)(
            overlap_distance
        )  # Zero the negative values, the distances can be larger than radius
        overlap_distance = overlap_distance.sum()

        # Calculate the collision count
        # collision_count = (
        #     collision_matrix.sum() - n
        # )  # Subtract n to remove self-collisions
        # # Calculate the penalty
        # penalty = collision_count / n

        return overlap_distance * collision_loss_coefficient

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

        # X distribution
        fake_xq = torch.quantile(fake_images[:, :, 0], quantiles)

        real_xq = torch.tensor(  # TODO: Make this a parameter
            [0.05, 0.25, 0.50, 0.75, 0.95], device=fake_images.device
        )  # NOTE: Tensors normalized from 0 to 1 so the quantiles should match
        # TODO: Test this

        # Y distribution
        fake_yq = torch.quantile(fake_images[:, :, 1], quantiles)
        real_yq = torch.tensor(  # TODO: Make this a parameter
            [0.05, 0.25, 0.50, 0.75, 0.95], device=fake_images.device
        )  # NOTE: Tensors normalized from 0 to 1 so the quantiles should match
        # TODO: Test this

        # Also do a xy loss

        # fake_xy = fake_images[:, :, 0] + fake_images[:, :, 1]
        # real_xy = real_images[:, :, 0] + real_images[:, :, 1]

        # fake_xyq = torch.quantile(fake_xy, quantiles)
        # real_xyq = torch.quantile(real_xy, quantiles)

        loss = (self.mse(fake_xq, real_xq) + self.mse(fake_yq, real_yq)) / 2

        return loss

    def forward(self, real_images, fake_images, fake_outputs, real_labels):

        loss = 0

        if self.radius_loss:
            self.prev_radius_loss = self._radius_loss(
                real_images=real_images, fake_images=fake_images
            )
            loss += self.prev_radius_loss

        if self.physical_feasibility_loss:
            self.prev_physical_feasibility_loss = self._physical_feasibility_loss(
                fake_images,
                collision_loss_coefficient=self.collision_loss_coefficient,  # TODO: Make this a parameter
            )
            loss += self.prev_physical_feasibility_loss

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


class CryinGANDiscriminatorLoss(nn.Module):
    def __init__(self, mu=10):
        super().__init__()
        self.mu = mu
        self.bce = BCELoss()

    def forward(
        self,
        real_outputs,
        fake_outputs,
        real_images,
        fake_images,
        discriminator,
    ):
        # Discriminator loss
        real_labels = torch.ones_like(real_outputs)
        fake_labels = torch.zeros_like(fake_outputs)

        d_loss = self.bce(real_outputs, real_labels) + self.bce(
            fake_outputs, fake_labels
        )

        # Gradient penalty term
        alpha = torch.rand(real_images.size(0), 1, 1).to(real_images.device)
        interpolates_coord = alpha * real_images + (1 - alpha) * fake_images
        interpolates_coord.requires_grad_(True)
        d_interpolates_coord = discriminator(interpolates_coord)

        grad_outputs_coord = torch.ones(d_interpolates_coord.size()).to(
            real_images.device
        )

        gradients = torch.autograd.grad(
            outputs=d_interpolates_coord,
            inputs=interpolates_coord,
            grad_outputs=grad_outputs_coord,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = self.mu * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        # Total loss
        total_loss = d_loss + gradient_penalty

        return total_loss


def build_loss_fn(**loss_config):

    name = loss_config.pop("name")

    module = importlib.import_module(".", package="src.models.losses")
    loss_fn_builder = getattr(module, name)

    loss_fn = loss_fn_builder(**loss_config)

    return loss_fn
