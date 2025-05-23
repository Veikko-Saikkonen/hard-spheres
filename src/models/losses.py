from copy import deepcopy
import torch
from torch import nn
import importlib

from torch.nn import BCELoss, MSELoss  # Enable by default
from pynndescent import NNDescent


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
        grid_order_loss,
        coefficients={
            "gan_loss": 1,
            "radius_loss": 1,
            "grid_density_loss": 1,
            "physical_feasibility_loss": 1,
            "distance_loss": 1,
            "grid_order_loss": 1,
            "grid_order_k":4
        },
        distance_loss=False,
    ):
        super().__init__()
        self.gan_loss = gan_loss
        self.gan_loss_fn = BCELoss()
        self.radius_loss = radius_loss
        self.grid_density_loss = grid_density_loss
        self.physical_feasibility_loss = physical_feasibility_loss
        self.distance_loss = distance_loss
        self.grid_order_loss = grid_order_loss
        self.coefficients = coefficients
        self.grid_order_k = self.coefficients["grid_order_k"]

        # For logging, save the results
        self.prev_gan_loss = torch.tensor([0])
        self.prev_radius_loss = torch.tensor([0])
        self.prev_grid_density_loss = torch.tensor([0])
        self.prev_physical_feasibility_loss = torch.tensor([0])
        self.prev_distance_loss = torch.tensor([0])
        self.prev_grid_order_loss = torch.tensor([0])

        self.mse = MSELoss()

    def _radius_loss(self, real_images, fake_images):
        # Loss based on sum of radiuses
        quantiles = torch.tensor(  # TODO: Make this a parameter
            [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            dtype=torch.float32,
            device=fake_images.device,
        )

        fake_rq = torch.quantile(fake_images[:, :, 2], quantiles)
        real_yq = torch.quantile(real_images[:, :, 2], quantiles)
        loss = self.mse(fake_rq, real_yq)

        return loss

    def _distance_loss(self, real_images, fake_images, k=3, eps=1e-6):
        # Loss based on the distribution of distances
        quantiles = torch.tensor(  # TODO: Make this a parameter
            [0.05, 0.5, 0.95],
            dtype=torch.float32,
            device=fake_images.device,
        )

        # dist_real = torch.cdist(
        #     real_images[:, :, :2], real_images[:, :, :2], p=2
        # )  # TODO: Try p=infinity
        # dist_fake = torch.cdist(fake_images[:, :, :2], fake_images[:, :, :2], p=2)
        dist_real = torch.sqrt(((real_images[:, :, :2].unsqueeze(2) - real_images[:, :, :2].unsqueeze(1)) ** 2).sum(-1) + eps)
        dist_fake = torch.sqrt(((fake_images[:, :, :2].unsqueeze(2) - fake_images[:, :, :2].unsqueeze(1)) ** 2).sum(-1) + eps)

        # Take the k nearest neighbours
        if isinstance(int(k), int) and k:
            dist_real = torch.topk(dist_real, k=k, dim=2, largest=False).values
            dist_fake = torch.topk(dist_fake, k=k, dim=2, largest=False).values

        # Take the lower triangle to avoid duplicates
        # mask = torch.tril(torch.ones_like(dist_real), diagonal=1)
        # dist_real = dist_real[mask == 1]
        # dist_fake = dist_fake[mask == 1]

        dist_real = dist_real.reshape(real_images.shape[0], -1)
        dist_fake = dist_fake.reshape(fake_images.shape[0], -1)

        real_yq = torch.quantile(dist_real, quantiles, dim=1)
        fake_rq = torch.quantile(dist_fake, quantiles, dim=1)

        loss = self.mse(fake_rq, real_yq)

        return loss
    
    # @staticmethod
    # def hexatic_order_parameter_batched(coords, k):

    #     # Ensure coords is shape (N, 2)
    #     assert coords.dim() == 3 and coords.size(2) == 2, \
    #         "coords must be of shape (B,N,2)."
        
    #     # Number of points
    #     N = coords.size(1)

    #     # 1) Compute pairwise differences: shape (N, N, 2)
    #     #    diffs[i, j, :] = coords[j] - coords[i]
    #     #    (the vector from i to j)
    #     diffs = coords.unsqueeze(2) - coords.unsqueeze(1)

    #     # 2) Compute the angles for each pair using atan2(y, x).
    #     #    angles[i, j] = angle of vector from i to j
    #     angles = torch.atan2(diffs[..., 1], diffs[..., 0])  # shape (N, N)

    #     # 3) Compute exp(i * 6 * theta_{ij}).
    #     #    We can leverage PyTorch's complex support:
    #     # e_i6theta = torch.exp(1j * 6 * angles) # 6 in the hex lattice
    #     e_i6theta = torch.exp(1j * k * angles) # Replaced with 4 in the square lattice

    #     # 4a) Typically, we do not include the j = i term in the sum (angle to itself).
    #     #    So we set the diagonal elements to zero.
    #     # diag_idx = torch.arange(N)
    #     # e_i6theta[diag_idx, diag_idx] = 0.0 + 0.0j

    #     # 4b) we want to ignore everything except the n nearest neighbors

    #     distances = torch.cdist(coords, coords, p=2)
    #     # Set the diagonal to inf as that is the distance to itself and we want to ignore that
    #     diag_idx = torch.arange(N)
    #     distances[:, diag_idx, diag_idx] = torch.inf
    #     distances_topk = torch.topk(distances, dim=1, k=k, largest=False).values.max(dim=1).values
    #     mask = (distances.transpose(0,1) > distances_topk).transpose(0,1) # This takes in the top k distances and sets everything else to zero
    #     e_i6theta[mask] = 0.0 + 0.0j
    #     # 5) Define Ni as the number of neighbors for each point i.
    #     # Count the number of neighbours per point from the mask

    #     Ni = mask.shape[1]-mask.sum(dim=2).unsqueeze(-2) # Ni = k
    #     # Ni = k
    #     # 6) Sum over j for each i and normalize by Ni.
    #     #    psi[i] = (1/Ni) * sum_{j != i} exp(i 6 theta_{ij})
    #     # psi = e_i6theta.sum(dim=1) / Ni
    #     psi = (e_i6theta / Ni).sum(dim=2)

    #     return psi
    
    @staticmethod
    def hexatic_order_parameter_batched(coords, k, sigma=None, eps=1e-6):
        """
        Computes the local hexatic order parameter for each point in a batch,
        with additional numerical stabilization to avoid exploding gradients.

        Args:
            coords: Tensor of shape (B, N, 2), where B is the batch size and N is the number of points.
            k:      Number of nearest neighbors used to define the kth-distance threshold.
                    (For a hexatic order parameter, k is typically 6.)
            sigma:  Softness parameter for the sigmoid weighting. If None, it is set to 0.1 times the kth distance.
            eps:    A small constant to prevent numerical instabilities.

        Returns:
            psi:    Tensor of shape (B, N) containing a complex number (the order parameter)
                    for each point.
        """
        B, N, _ = coords.shape

        # 1. Compute pairwise differences: shape (B, N, N, 2)
        diffs = coords.unsqueeze(2) - coords.unsqueeze(1)
        
        # 2. Stabilize the x-component: if nearly zero, add eps.
        diff_x = diffs[..., 0]
        # Create a mask where diff_x is too small (absolute value below eps)
        near_zero = diff_x.abs() < eps
        diff_x = diff_x + eps * near_zero.float()  # add eps where needed
        
        # 3. Compute angles using the stabilized x-component: shape (B, N, N)
        angles = torch.atan2(diffs[..., 1], diff_x)
        
        # 4. Compute exp(i * k * theta) for each pair: shape (B, N, N)
        e_iktheta = torch.exp(1j * k * angles)
        
        # 5. Compute pairwise distances manually (Euclidean) with eps for stability
        distances = torch.sqrt(((coords.unsqueeze(2) - coords.unsqueeze(1)) ** 2).sum(-1) + eps)
        
        # Set self-distances to infinity so that a point is never its own neighbor.
        diag_mask = torch.eye(N, dtype=torch.bool, device=coords.device).unsqueeze(0).expand(B, N, N)
        distances = distances.masked_fill(diag_mask, float('inf'))
        
        # 6. For each point, compute the kth smallest distance (neighbor threshold)
        kth_distance = torch.topk(distances, k, dim=2, largest=False).values[..., -1].detach()  # shape (B, N)
        
        # 7. Define sigma if not provided, and ensure sigma is not too small.
        if sigma is None:
            sigma = 0.1 * kth_distance.unsqueeze(-1).clamp(min=eps)  # shape (B, N, 1)
        else:
            sigma = sigma * torch.ones_like(kth_distance.unsqueeze(-1))
        sigma = sigma.clamp(min=eps)
        
        # 8. Compute soft weights for neighbor contributions:
        #    Use the sigmoid on a clamped argument to avoid overflow.
        arg = (kth_distance.unsqueeze(-1) - distances) / sigma
        arg = arg.clamp(min=-50, max=50)  # avoid extremely large exponents
        weights = torch.sigmoid(arg)      # shape (B, N, N)
        
        # 9. Compute a weighted sum and normalize by the sum of weights.
        weighted_sum = (e_iktheta * weights).sum(dim=2)  # shape (B, N)
        weights_sum = weights.sum(dim=2).clamp(min=eps)    # shape (B, N)
        psi = weighted_sum / weights_sum

        return psi
    

    def _grid_order_loss(self, samples):

        xy = samples.clone()
        # Compute the hexatic order parameter
        psi_values = self.hexatic_order_parameter_batched(xy, k=self.grid_order_k)

        # Compute the loss as the sum of the imaginary parts
        loss = -torch.mean(torch.sqrt(psi_values * psi_values.conj())).real
        return loss

    def _physical_feasibility_loss(self, fake_points):

        # Rescale the coordinates and radii back to their original scales
        x = fake_points[:, :, 0]
        y = fake_points[:, :, 1]
        radii = fake_points[:, :, 2]

        # TODO: Scale

        # Combine rescaled values into one tensor
        rescaled_points = torch.stack((x, y, radii), dim=2)
        # If the distance is less than the sum of the radii, then it's a collision

        # Calculate the pairwise distance matrix
        dist = torch.cdist(rescaled_points[:, :, :2], rescaled_points[:, :, :2])
        dist = dist.tril(diagonal=0)
        dist[dist <= 0] = (
            torch.inf
        )  # Set the diagonal (=distance to self) to infinity to avoid loss due to that

        # Calculate the sum of the radii of the two points
        radii = (
            rescaled_points[:, :, 2].unsqueeze(1).abs()
            + rescaled_points[:, :, 2].unsqueeze(2).abs()
        )
        epsilon = 1e-4
        # Instead of a hard limit, use a differentiable loss function
        overlap_distance = radii - (
            dist + epsilon
        )  # epsilon = tolerance for overlapping
        overlap_distance = nn.ReLU(inplace=False)(
            overlap_distance
        )  # Zero the negative values, the distances can be larger than radius
        # Square the result to emphasize larger overlaps
        # Take the root to make the loss differentiable
        overlap_distance = torch.sqrt(overlap_distance**2)  #
        overlap_distance = overlap_distance.sum() / (
            radii.sum() / 2
        )  # How much of the total radius is overlapping

        return overlap_distance

    def _gan_loss(self, fake_outputs, real_labels):
        return self.gan_loss_fn(fake_outputs, real_labels)

    @staticmethod
    def _non_saturating_gan_loss(fake_outputs):
        return -torch.mean(
            torch.log(fake_outputs)
        )  # Maximize the probability of the fake samples

    def _grid_density_loss(self, real_images, fake_images):
        quantiles = torch.tensor(
            # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.05, 0.25, 0.50, 0.75, 0.95],  #
            dtype=torch.float32,
            device=fake_images.device,
        )
        # first two columns are x and y coordinates
        # Count the 5, 25, 50, 75, 95 quantiles for the first two columns and compare to the ground truth

        # X distribution
        fake_xq = torch.quantile(fake_images[:, :, 0], quantiles)
        real_xq = torch.quantile(real_images[:, :, 0], quantiles)

        # Y distribution
        fake_yq = torch.quantile(fake_images[:, :, 1], quantiles)
        real_yq = torch.quantile(real_images[:, :, 1], quantiles)

        loss = (self.mse(fake_xq, real_xq) + self.mse(fake_yq, real_yq)) / 2

        return loss

    def forward(self, real_images, fake_images, fake_outputs):
        """Calculate the loss for the generator.
        NOTES:
        - If loss coefficient is 0, the loss is not calculated, otherwise it is calculated and tracked
        - If the loss is calculated, it is multiplied by the coefficient
        - If the loss is set to False, it is not added to the total loss (even if the coefficient is not 0)

        Args:
            real_images (_type_): _description_
            fake_images (_type_): _description_
            fake_outputs (_type_): _description_

        Returns:
            _type_: _description_
        """

        loss = 0

        if self.radius_loss:
            self.prev_radius_loss = (
                self._radius_loss(real_images=real_images, fake_images=fake_images)
                * self.coefficients["radius_loss"]
            )
            loss += self.prev_radius_loss

        if self.coefficients["physical_feasibility_loss"]:
            # self.prev_physical_feasibility_loss = (
            #     nn.ReLU()(  # ReLU to avoid negative values in case of no overlap
            #         self._physical_feasibility_loss(
            #             fake_images,
            #         )
            #         - self._physical_feasibility_loss(
            #             real_images,
            #         )  # Compare to the real image to correct the scale
            #     )
            #     * self.coefficients["physical_feasibility_loss"]
            # )
            self.prev_physical_feasibility_loss = self._physical_feasibility_loss(
                        fake_images,
                    ) * self.coefficients["physical_feasibility_loss"]
            
        if self.physical_feasibility_loss:
            loss += self.prev_physical_feasibility_loss

        if self.gan_loss:
            real_labels = torch.ones_like(fake_outputs) - 0.1
            self.prev_gan_loss = (
                self._gan_loss(fake_outputs, real_labels)
                * self.coefficients["gan_loss"]
            )
            # self.prev_gan_loss = (
            #     self._non_saturating_gan_loss(fake_outputs)
            #     * self.coefficients["gan_loss"]
            # )

            loss += self.prev_gan_loss

        if self.coefficients["grid_density_loss"]:
            self.prev_grid_density_loss = (
                self._grid_density_loss(
                    real_images=real_images, fake_images=fake_images
                )
                * self.coefficients["grid_density_loss"]
            )
        if self.grid_density_loss:
            loss += self.prev_grid_density_loss

        if self.coefficients["distance_loss"]:
            self.prev_distance_loss = (
                self._distance_loss(real_images=real_images, fake_images=fake_images)
                * self.coefficients["distance_loss"]
            )
        if self.distance_loss:
            loss += self.prev_distance_loss

        if self.coefficients["grid_order_loss"]:
            self.prev_grid_order_loss = self._grid_order_loss(fake_images) * self.coefficients["grid_order_loss"]
        if self.grid_order_loss:
            loss += self.prev_grid_order_loss

        return loss


class CryinGANDiscriminatorLoss(nn.Module):
    def __init__(self, mu=10):
        super().__init__()
        self.mu = mu
        self.bce = BCELoss()
        self.prev_gradient_penalty = torch.tensor([0])

    def gradient_penalty(self, real_images, fake_images, discriminator):
        # Gradient penalty term
        alpha = torch.rand(real_images.size(0), 1, 1).to(real_images.device)
        interpolates_coord = alpha * real_images + (1 - alpha) * fake_images
        interpolates_coord = interpolates_coord.requires_grad_(True)
        d_interpolates_coord = discriminator(interpolates_coord)

        grad_outputs_coord = torch.ones_like(d_interpolates_coord)

        gradients = torch.autograd.grad(
            outputs=d_interpolates_coord,
            inputs=interpolates_coord,
            grad_outputs=grad_outputs_coord,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            is_grads_batched=False,
        )[0]

        gradient_penalty = (
            self.mu * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        )  # TODO: Check if this is correct
        return gradient_penalty

    def forward(
        self,
        real_outputs,  # Discriminator outputs for real images
        fake_outputs,  # Discriminator outputs for fake images
        real_images,  # Real images
        fake_images,  # Generator outputs
        discriminator,  # Discriminator model
    ):
        # Discriminator loss
        real_labels = torch.ones_like(real_outputs) - 0.1
        fake_labels = torch.zeros_like(fake_outputs) + 0.1

        self.prev_gan_loss = self.bce(real_outputs, real_labels) + self.bce(
            fake_outputs, fake_labels
        )

        # Total loss
        self.prev_gradient_penalty = self.gradient_penalty(
            real_images=real_images,
            fake_images=fake_images,
            discriminator=discriminator,
        )
        total_loss = self.prev_gan_loss + self.prev_gradient_penalty

        return total_loss


def build_loss_fn(**loss_config):

    name = loss_config.pop("name")

    module = importlib.import_module(".", package="src.models.losses")
    loss_fn_builder = getattr(module, name)

    loss_fn = loss_fn_builder(**loss_config)

    return loss_fn
