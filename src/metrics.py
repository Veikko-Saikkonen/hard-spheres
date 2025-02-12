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

