import torch


def get_descriptors(data, experiment, descriptors=["phi"]):

    descriptors = []

    phi = float(experiment.split("-")[-1])
    # Assume future data will have more dimensions
    phi = [[phi]] * 3
    descriptors.append(phi)

    # Convert to tensor
    descriptors = torch.tensor(descriptors, dtype=torch.float32)

    # For future compatibility, even if using a single descriptor,
    # we will keep it as a 2D tensor
    if len(descriptors.shape) == 1:
        descriptors = descriptors.unsqueeze(1)

    # In the future descriptors may be tensors instead of scalars, include in design
    if len(descriptors.shape) == 2:
        descriptors = descriptors.unsqueeze(2)

    # descriptors = descriptors.flatten(start_dim=1)

    descriptors = descriptors.transpose(-1, -2)

    return descriptors


# Load data for the GAN
