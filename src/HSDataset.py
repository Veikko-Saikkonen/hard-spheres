import torch

from torch.utils.data import TensorDataset
import lightning as L

from .load_data import get_descriptors


# Create a dataset
class HSDataset(TensorDataset):
    def __init__(
        self,
        dataframe,
        device="cpu",
        descriptor_list=["phi"],
        synthetic_samples=False,
        downsample=0,
    ):
        print("Creating Dataset")
        print("Descriptor List: ", descriptor_list)
        # Split each experiment and sample into a separate sample

        descriptors = []
        samples = []

        for (
            experiment,
            sample,
        ) in dataframe.index.unique():  # TODO: Check if this is the right way to do it
            sample = torch.tensor(
                dataframe.loc[(experiment, sample), :].copy().values,
                dtype=torch.float32,
                device=device,
            )
            sample_descriptors = get_descriptors(
                sample, experiment, descriptors=descriptor_list
            )

            samples.append(
                sample.unsqueeze(0)
            )  # We are creating a 3D tensor of 2D samples
            descriptors.append(sample_descriptors)

        if synthetic_samples:

            if synthetic_samples["rotational"]:

                # Create a new samples by rotating the original sample, ie switching x and y
                rotated_samples = []
                for sample in samples:
                    new_sample = sample.clone()
                    new_sample[:, :, 0] = sample[:, :, 1]
                    new_sample[:, :, 1] = sample[:, :, 0]
                    rotated_samples.append(new_sample)
                samples += rotated_samples
                descriptors += descriptors

            if synthetic_samples["spatial_offset_static"]:
                # Create synthetic samples by adding noise to the original samples
                new_descriptors = []
                new_samples = []

                # Add noise to the samples
                # Amount of noise is in the synthetic_samples["spatial_offset_static"]

                noise = synthetic_samples["spatial_offset_static"]

                # Move the whole sample in a random direction
                for sample in samples:
                    new_sample = sample.clone()
                    direction = torch.randint(
                        0, 4, (1,)
                    )  # 0: Up, 1: Down, 2: Left, 3: Right

                    # Move the sample in the direction of the noise
                    # NOTE: This is a very naive way of adding noise
                    if direction == 0:
                        new_sample[:, :, 1] += noise
                    elif direction == 1:
                        new_sample[:, :, 1] -= noise
                    elif direction == 2:
                        new_sample[:, :, 0] -= noise
                    elif direction == 3:
                        new_sample[:, :, 0] += noise

                    new_samples.append(new_sample)

                samples += new_samples
                descriptors += descriptors

            if synthetic_samples["shuffling"]:  # Shuffle in the end
                # Create synthetic samples by shuffling the original samples
                shuffled_samples = []
                new_descriptors = []
                for i in range(
                    synthetic_samples["shuffling"]
                ):  # Add one permutation for each shuffle
                    for sample in samples:
                        shuffled_samples.append(sample[torch.randperm(sample.size(0))])
                    new_descriptors += descriptors

                samples += shuffled_samples
                descriptors += new_descriptors

        if downsample:
            new_samples = []
            for sample in samples:
                downsample_integer = int(sample.shape[1] * downsample)
                # Random pick
                # pick_index = torch.randperm(sample.shape[1])[:downsample_integer]
                # Pick the ones closest to the center
                center = torch.tensor([0.0, 0.0], device=device)
                distance = torch.norm(sample[:, :, :2] - center, dim=0)
                _, pick_index = distance.topk(downsample_integer, largest=False, dim=0)
                pick_index = slice(0, downsample_integer)
                _sample = sample[:, pick_index]
                new_samples.append(_sample)
            samples = new_samples

        self.x = torch.concat(descriptors)  # Descriptors are the input.
        self.y = torch.concat(samples)  # Sample point cloud is the target
        self.samples = samples

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        pointcloud = self.x[idx]
        descriptors = self.y[idx]
        return pointcloud, descriptors

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self
