from torch import nn
from torch import rand, randn


class Generator(nn.Module):
    def __init__(
        self, kernel_size, stride, rand_features=64, out_dimensions=3, out_samples=2000
    ):
        super().__init__()

        self.rand_features = rand_features
        out_features = out_samples * out_dimensions

        self.model = nn.Sequential(
            # TODO: Add this section to take descriptors as input instead of random noise
            # nn.Flatten(1, -1),
            # nn.Linear(3, in_features), # 33k
            # nn.ReLU(True),
            ## First fully connected layer
            # nn.Linear(in_features, 128 * 264), # 33k
            nn.Linear(
                in_features=rand_features, out_features=out_features
            ),  # 12k, half size
            nn.ReLU(True),
            # # Reshape to (128, 264, 1, 1)
            # nn.Unflatten(1, (128, 264, 1)),
            nn.Unflatten(1, (out_dimensions, out_samples, 1)),
            # # Transposed Convolution 1
            nn.ConvTranspose2d(
                out_dimensions, 256, kernel_size=kernel_size, stride=(1, 3), padding=0
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # # # Transposed Convolution 2
            nn.ConvTranspose2d(
                256, 512, kernel_size=kernel_size, stride=stride, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # # Transposed Convolution 3
            nn.ConvTranspose2d(
                512, 256, kernel_size=kernel_size, stride=stride, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # # Transposed Convolution 4 to get to 3 channels, 2000 samples and 1 feature
            nn.ConvTranspose2d(
                256, out_dimensions, kernel_size=(1, 1), stride=(1, 1), padding=1
            ),
            nn.Flatten(1, -1),
            nn.Unflatten(1, (out_samples, out_dimensions)),
            nn.Sigmoid(),
        )

        for layer in self.model:
            if hasattr(layer, "weight"):
                nn.init.uniform_(layer.weight, a=-0.33, b=0.33)

            if hasattr(layer, "bias"):
                nn.init.uniform_(layer.bias, a=-0.1, b=0.1)

    def generate_noise(self, batch_size):
        return rand(
            (batch_size, self.rand_features), device=self.model[0].weight.device
        )

    def forward(self, x):
        # TODO: Use x in the model instead of purely random noise

        noise = self.generate_noise(x.size(0))

        return self.model(noise)


class Discriminator2D(nn.Module):
    def __init__(self, input_channels, in_samples=2000):
        super().__init__()
        if in_samples % 16 != 0:
            raise ValueError("in_samples must be divisible by 16")
        self.main = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Unflatten(1, (input_channels, in_samples)),
            # Unflatten from 3x2000 to 3x125x16
            nn.Unflatten(2, (in_samples // 16, 16)),
            # Convolutional layer 1
            nn.Conv2d(input_channels, 256, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # # Convolutional layer 2
            nn.Conv2d(256, 256, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # # Convolutional layer 3
            nn.Conv2d(256, 256, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # # Average Pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1, -1),
        )

        self.fc_layers = nn.Sequential(
            # Fully connected layers
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        conv_output = self.main(input)
        conv_output_flat = conv_output.view(conv_output.size(0), -1)  # Flatten
        output = self.fc_layers(conv_output_flat)
        return output


class CCCGDiscriminator(nn.Module):
    # Inspired by: https://dl-acm-org.libproxy.aalto.fi/doi/abs/10.1145/3532213.3532218
    def __init__(self, input_channels, in_samples=2000):
        super().__init__()

        self.main = nn.Sequential(
            # Convolutional layer 2
            nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # # # Convolutional layer 2
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # Convolutional layer 3
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.AvgPool2d((500, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1, -1),
        )

        self.fc_layers = nn.Sequential(
            # NOTE: This may not be the best way to reduce the size of the tensor
            # Fully connected layers
            nn.Linear(256, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        # First convolve each atom individually
        x = input.unsqueeze(-1).transpose(1, 2)
        # Then each atom is concatenated to the next one
        conv_output = self.main(x)
        # Finally flatten and pass through fully connected layers
        conv_output_flat = conv_output.view(conv_output.size(0), -1)
        output = self.fc_layers(conv_output_flat)
        return output


class CCCGenerator(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        rand_features=513,
        out_dimensions=3,
        out_samples=2000,
        latent_dim=6,
    ):

        # Inspired by: https://dl-acm-org.libproxy.aalto.fi/doi/abs/10.1145/3532213.3532218

        super().__init__()

        self.rand_features = rand_features
        latent_dim = 6
        # latent_features = 256 * 40  # From the paper, samples x latent_dim
        latent_features = out_samples * latent_dim  # Samples x latent_dim

        self.model = nn.Sequential(
            ## First fully connected layer
            # nn.Linear(in_features, 128 * 264), # 33k
            nn.Linear(
                in_features=rand_features, out_features=latent_features
            ),  # 12k, half size
            nn.ReLU(True),
            # # Reshape to (128, 264, 1, 1)
            # nn.Unflatten(1, (128, 264, 1)),
            nn.Unflatten(1, (latent_dim, out_samples, 1)),
            # # Transposed Convolution 1
            nn.ConvTranspose2d(
                latent_dim, 256, kernel_size=(1, out_dimensions), stride=1, padding=0
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # # # Transposed Convolution 2
            nn.ConvTranspose2d(256, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # # Transposed Convolution 3
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # # Transposed Convolution 4 to get to 3 channels, 2000 samples and 1 feature
            nn.ConvTranspose2d(64, 1, kernel_size=(1, 1), stride=1, padding=0),
            nn.Sigmoid(),
            nn.Flatten(1, 2),
        )

        # for layer in self.model: # TODO: check on this later
        #     if hasattr(layer, "weight"):
        #         nn.init.uniform_(layer.weight, a=-0.33, b=0.33)

        #     if hasattr(layer, "bias"):
        #         nn.init.uniform_(layer.bias, a=-0.1, b=0.1)

    def generate_noise(self, batch_size):
        return rand(
            (batch_size, self.rand_features), device=self.model[0].weight.device
        )

    def forward(self, x):
        # TODO: Use x in the model instead of purely random noise

        noise = self.generate_noise(x.size(0))

        return self.model(noise)
