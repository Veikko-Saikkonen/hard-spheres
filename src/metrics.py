"""
Metrics for measuring the performance of the models.
"""

import torch

def packing_fraction(samples, fix_r=None, box_size=1, batch=False):
    """
    Calculate the packing fraction of the samples.

    Args:
        samples (torch.tensor): The samples to calculate the packing fraction for. Samples need to have x, y, and radius (unless fix_r is provided) as the last three dimensions.
        fix_r (float, optional): If provided, the radius to use for all samples. Defaults to None.
        box_size (float, optional): The size of the box to calculate the packing fraction in. Defaults to 1.
        batch (bool, optional): Whether the samples are batched. Defaults to False.

    Returns:
        float: The packing fraction.
    """

    # TODO: Replace with the pixel based variant.

    # Calculate the packing fraction
    if batch:
        # Calculate the packing fraction for each sample in the batch
        return torch.tensor(
            [packing_fraction(samples[i], fix_r, box_size) for i in range(samples.shape[0])]
            ).mean()

    total_area = box_size ** 2

    if fix_r is not None:
        # Use the same radius for all samples
        radii = torch.ones(samples.shape[0]) * fix_r
    else:
        radii = samples[:, 2]

    total_area -= torch.pi * torch.sum(radii ** 2)

    return 1 - total_area / (box_size ** 2)

import torch
import math
import matplotlib.pyplot as plt

def packing_fraction_pixel(samples, total_area, r_fix=None, resolution=128, return_grid=False):
    """
    Compute the pixel-based packing fraction for batched point clouds of circles using PyTorch.
    
    Parameters:
      samples (torch.Tensor): A tensor of shape (B, N, 3) with [x, y, r] if r_fix is None,
                              or shape (B, N, 2) with [x, y] if a fixed radius is provided.
                              If a non-batched tensor of shape (N, 3) or (N, 2) is given,
                              it will be unsqueezed to add a batch dimension.
      total_area (float): Total area of the domain. The function assumes the domain is a square 
                          with side length sqrt(total_area).
      r_fix (float, optional): If provided, all circles are drawn with this fixed radius (overrides any r in samples).
      resolution (int, optional): Number of pixels along each axis in the grid.
      return_grid (bool, optional): If True, returns a tuple (pf, union_mask, xs, ys) where:
                                    - pf is a tensor of shape (B,) containing the packing fraction for each batch,
                                    - union_mask is a boolean tensor of shape (B, resolution, resolution) with the per-batch pixel coverage,
                                    - xs and ys are the grid coordinate tensors.
    
    Returns:
      If return_grid is False:
          torch.Tensor: A tensor of shape (B,) containing the packing fraction for each batch element.
      If return_grid is True:
          tuple: (pf, union_mask, xs, ys)
    """
    # If samples is not batched, add a batch dimension.
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)
    
    B, N, k = samples.shape
    device = samples.device
    dtype = samples.dtype

    # Determine the side length of the square domain.
    side = math.sqrt(total_area)
    
    # Create grid coordinates over the domain.
    xs = torch.linspace(0, side, resolution, device=device, dtype=dtype)
    ys = torch.linspace(0, side, resolution, device=device, dtype=dtype)
    
    # Create a meshgrid of pixel centers.
    try:
        xv, yv = torch.meshgrid(xs, ys, indexing='xy')
    except TypeError:
        # For older PyTorch versions without the 'indexing' argument.
        xv, yv = torch.meshgrid(xs, ys)
    # xv and yv are of shape (resolution, resolution)

    # Extract x and y coordinates for the circle centers.
    x_centers = samples[..., 0]  # shape: (B, N)
    y_centers = samples[..., 1]  # shape: (B, N)
    
    # Determine the radii.
    if r_fix is not None:
        # Create a tensor of shape (B, N) filled with the fixed radius.
        r = torch.full((B, N), r_fix, device=device, dtype=dtype)
    else:
        if k < 3:
            raise ValueError("samples should have at least 3 columns (x, y, r) if r_fix is not provided")
        r = samples[..., 2]  # shape: (B, N)
    
    # Reshape circle parameters for broadcasting.
    # Each is reshaped to (B, N, 1, 1) so that we can subtract from the grid of shape (1, 1, resolution, resolution).
    x_centers_exp = x_centers.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    y_centers_exp = y_centers.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    r_exp = r.unsqueeze(-1).unsqueeze(-1)                  # (B, N, 1, 1)
    
    # Expand the grid to have shape (1, 1, resolution, resolution)
    xv_exp = xv.unsqueeze(0).unsqueeze(0)  # (1, 1, resolution, resolution)
    yv_exp = yv.unsqueeze(0).unsqueeze(0)  # (1, 1, resolution, resolution)
    
    # Compute the squared distance from every pixel to every circle center.
    # The result is of shape (B, N, resolution, resolution).
    dist_sq = (xv_exp - x_centers_exp)**2 + (yv_exp - y_centers_exp)**2
    
    # Create a boolean mask indicating which pixels are inside each circle.
    mask = dist_sq <= (r_exp**2)  # shape: (B, N, resolution, resolution)
    
    # Compute the union over circles for each batch (logical OR over the circle dimension).
    union_mask = mask.any(dim=1)  # shape: (B, resolution, resolution)
    
    # Calculate the area of each pixel.
    pixel_area = (side / resolution) ** 2
    
    # Compute the union area per batch by summing the True values and multiplying by the pixel area.
    # The sum is computed over the last two dimensions (pixels).
    union_area = union_mask.sum(dim=[1, 2]).to(dtype) * pixel_area  # shape: (B,)
    
    # Compute the packing fraction per batch.
    pf = union_area / total_area  # shape: (B,)
    
    if return_grid:
        return pf, union_mask, xs, ys
    else:
        return pf
    

import torch
import math
import matplotlib.pyplot as plt

def packing_fraction_pixel_memory_friendly(samples, total_area, r_fix=None, resolution=512, return_grid=False):
    """
    Compute the pixel-based packing fraction for batched point clouds of circles using PyTorch
    in a memory-friendly manner. Instead of creating an intermediate tensor of shape
    (B, N, resolution, resolution), it iteratively "paints" each circle onto an accumulator.

    Parameters:
      samples (torch.Tensor): A tensor of shape (B, N, 3) with [x, y, r] if r_fix is None,
                              or shape (B, N, 2) with [x, y] if a fixed radius is provided.
                              If a non-batched tensor of shape (N, 3) or (N, 2) is given,
                              it will be unsqueezed to add a batch dimension.
      total_area (float): Total area of the domain (assumed to be a square with side sqrt(total_area)).
      r_fix (float, optional): If provided, all circles are drawn with this fixed radius (overrides any r in samples).
      resolution (int, optional): Number of pixels along each axis in the grid.
      return_grid (bool, optional): If True, returns a tuple (pf, union_mask, xs, ys) where:
                                    - pf is a tensor of shape (B,) with the packing fraction for each batch element,
                                    - union_mask is a boolean tensor of shape (B, resolution, resolution)
                                      indicating the pixel coverage,
                                    - xs and ys are the 1D grid coordinate tensors.

    Returns:
      If return_grid is False:
          torch.Tensor: A tensor of shape (B,) containing the packing fraction for each batch element.
      If return_grid is True:
          tuple: (pf, union_mask, xs, ys)
    """
    # Ensure the input is batched.
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)  # Now shape is (1, N, K)
    
    B, N, k = samples.shape
    device = samples.device
    dtype = samples.dtype

    # Determine the side length of the square domain.
    side = math.sqrt(total_area)

    # Create grid coordinates.
    xs = torch.linspace(0, side, resolution, device=device, dtype=dtype)
    ys = torch.linspace(0, side, resolution, device=device, dtype=dtype)
    try:
        xv, yv = torch.meshgrid(xs, ys, indexing='xy')
    except TypeError:
        xv, yv = torch.meshgrid(xs, ys)
    # xv and yv have shape (resolution, resolution)

    # Initialize the union mask for each batch element.
    union_mask = torch.zeros((B, resolution, resolution), dtype=torch.bool, device=device)

    # Extract circle center coordinates.
    x_centers = samples[..., 0]  # shape: (B, N)
    y_centers = samples[..., 1]  # shape: (B, N)

    # Determine radii.
    if r_fix is not None:
        r = torch.full((B, N), r_fix, device=device, dtype=dtype)
    else:
        if k < 3:
            raise ValueError("samples should have at least 3 columns (x, y, r) if r_fix is not provided")
        r = samples[..., 2]  # shape: (B, N)

    # Loop over circles to update the union mask incrementally.
    # This loop avoids creating a large (B, N, resolution, resolution) tensor.
    for i in range(N):
        # Extract the parameters for the i-th circle in each batch.
        # Reshape to (B, 1, 1) for broadcasting.
        x_center_i = x_centers[:, i].view(B, 1, 1)
        y_center_i = y_centers[:, i].view(B, 1, 1)
        r_i = r[:, i].view(B, 1, 1)
        
        # Compute the squared distance from each pixel to the circle's center.
        # (xv, yv) has shape (resolution, resolution), broadcasting to (B, resolution, resolution)
        dist_sq = (xv - x_center_i)**2 + (yv - y_center_i)**2
        
        # Create the mask for pixels inside the current circle.
        circle_mask = dist_sq <= (r_i**2)
        
        # Update the union mask (logical OR).
        union_mask |= circle_mask

    # Compute the area of one pixel.
    pixel_area = (side / resolution) ** 2

    # Compute the union area per batch.
    union_area = union_mask.sum(dim=[1, 2]).to(dtype) * pixel_area

    # Compute the packing fraction.
    pf = union_area / total_area  # Shape: (B,)

    if return_grid:
        return pf, union_mask, xs, ys
    else:
        return pf

# --------------------------
# Example usage with visualization:
# --------------------------
if __name__ == '__main__':
    # Create example batched input.
    # Here we assume two batch elements.
    # Each batch element contains 3 circles (if a batch element has fewer circles,
    # pad with dummy circles that have zero radius).
    samples_batch = torch.tensor([
        # Batch element 0: 3 circles with variable radii.
        [
            [50.0, 50.0, 10.0],
            [80.0, 80.0, 15.0],
            [30.0, 70.0, 12.0]
        ],
        # Batch element 1: 2 circles; pad with a dummy circle.
        [
            [20.0, 20.0, 8.0],
            [60.0, 40.0, 10.0],
            [0.0,  0.0,  0.0]  # Dummy circle (r=0) will not contribute.
        ]
    ], dtype=torch.float32)
    
    # Define the total domain area (e.g., a 100 x 100 square has area 10000).
    total_area = 10000.0

    # --- Option 1: Using variable radii ---
    pf_batch, union_mask_batch, xs, ys = packing_fraction_pixel(
        samples_batch, total_area, r_fix=None, resolution=32, return_grid=True)
    print("Packing fraction for each batch (variable radii):", pf_batch)
    
    # Visualization: plot the union mask for each batch element.
    side = math.sqrt(total_area)
    for b in range(union_mask_batch.shape[0]):
        plt.figure(figsize=(6, 6))
        plt.imshow(union_mask_batch[b].cpu().numpy(),
                   extent=[0, side, 0, side],
                   origin='lower',
                   cmap='viridis')
        plt.title(f'Batch {b}: Coverage Grid (Variable Radii)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='Pixel Covered (True=1, False=0)')
        plt.show()
    
    # --- Option 2: Using a fixed radius ---
    # For fixed radius, samples need only supply x and y, so we use the first two columns.
    samples_batch_xy = samples_batch[..., :2]  # shape: (B, N, 2)
    pf_batch_fixed, union_mask_batch_fixed, xs_fixed, ys_fixed = packing_fraction_pixel(
        samples_batch_xy, total_area, r_fix=10, resolution=512, return_grid=True)
    print("Packing fraction for each batch (fixed radius 10):", pf_batch_fixed)
    
    # Visualization for fixed radius.
    for b in range(union_mask_batch_fixed.shape[0]):
        plt.figure(figsize=(6, 6))
        plt.imshow(union_mask_batch_fixed[b].cpu().numpy(),
                   extent=[0, side, 0, side],
                   origin='lower',
                   cmap='viridis')
        plt.title(f'Batch {b}: Coverage Grid (Fixed Radius 10)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='Pixel Covered (True=1, False=0)')
        plt.show()
