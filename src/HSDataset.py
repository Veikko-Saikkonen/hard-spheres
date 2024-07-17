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
                    new_sample[:, 0] = sample[:, 1]
                    new_sample[:, 1] = sample[:, 0]
                    rotated_samples.append(new_sample)
                samples += rotated_samples
                descriptors += descriptors

            if synthetic_samples["shuffling"]:
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
                pick_index = torch.randperm(sample.shape[1])[:downsample_integer]
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


class HSDatasetLighting(L.LightningDataModule):
    def __init__(
        self,
        dataframe,
        device="cpu",
        descriptor_list=["phi"],
        synthetic_samples=False,
        batch_size=32,
    ):
        super().__init__()
        self.dataframe = dataframe
        self.device = device
        self.descriptor_list = descriptor_list
        self.synthetic_samples = synthetic_samples
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = HSDataset(
                self.dataframe,
                device=self.device,
                descriptor_list=self.descriptor_list,
                synthetic_samples=self.synthetic_samples,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True
        )
