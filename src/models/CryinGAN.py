from torch import nn
from torch import rand, randn
import torch
import sys
import numpy as np
from mlflow.types import Schema, TensorSpec


class Generator(nn.Module):
    def __init__(
        self,
        kernel_size,
        first_layer_kernel_size,
        stride,
        rand_features=64,
        out_dimensions=3,
        out_samples=2000,
        channel_coefficient=1,
        clip_output: tuple = False,
        fix_r=None,
        r_options=[],
    ):
        super().__init__()
        # paper: https://arxiv.org/pdf/2404.06734

        self.rand_features = rand_features
        out_features = int(out_samples * (out_dimensions) / 2)

        self.fix_r = fix_r
        self.out_samples = out_samples
        self.out_dimensions = out_dimensions

        self.r_options = r_options
        self.n_r_options = len(r_options)
        self.clip_output = clip_output

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
            # nn.Unflatten(1, (out_dimensions, out_samples, 1)),
            nn.Unflatten(1, (-1, out_samples)),
            # # Transposed Convolution 1
            nn.ConvTranspose2d(
                out_dimensions,
                32 * channel_coefficient,
                kernel_size=first_layer_kernel_size,
                stride=(1, 3),  # NOTE: why stride this way?
                padding=0,
            ),
            nn.BatchNorm2d(32 * channel_coefficient),
            nn.ReLU(True),
            # # # Transposed Convolution 2
            nn.ConvTranspose2d(
                32 * channel_coefficient,
                16 * channel_coefficient,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            ),
            nn.BatchNorm2d(16 * channel_coefficient),
            nn.ReLU(True),
            # # Transposed Convolution 3
            nn.ConvTranspose2d(
                16 * channel_coefficient,
                8 * channel_coefficient,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            ),
            nn.BatchNorm2d(8 * channel_coefficient),
            nn.ReLU(True),
            # # Transposed Convolution 4 to get to 'out_dimensions' channels, 'out_samples' samples and 1 feature
            nn.ConvTranspose2d(
                8 * channel_coefficient,
                out_dimensions,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=0,
            ),
            nn.Flatten(1, -1),
            nn.Unflatten(1, (-1, out_dimensions)),
            nn.Sigmoid(),
        )

        for layer in self.model:  # TODO: check on this later
            if hasattr(layer, "weight"):
                nn.init.uniform_(layer.weight, a=-0.3, b=0.3)

            if hasattr(layer, "bias"):
                nn.init.uniform_(layer.bias, a=-0.15, b=0.15)

    def _get_clip_parameters(self):
        clip_output_min = torch.as_tensor(
            self.clip_output[0], device=self.model[0].weight.device
        )
        clip_output_max = torch.as_tensor(
            self.clip_output[1], device=self.model[0].weight.device
        )
        return clip_output_min, clip_output_max

    def generate_noise(self, batch_size):
        return rand(
            (batch_size, self.rand_features),
            device=self.model[0].weight.device,
            requires_grad=False,
        )

    def _softmax_radius_dimensions(self, x: torch.FloatTensor):

        # After the model has created the output, do softmax over the last dimensions to set radius

        r = torch.softmax(x[:, :, -self.n_r_options :])

        # Now we have [batch, n_samples, n_categories]
        # Map to a scalar radius
        # self.radius_options has options like [1,2,3], corresponding to one hot

        r = self.r_options * r

        x = torch.stack([x[:, :, : self.out_dimensions - 1], r], 0)

        return x

    def forward(self, x):
        # TODO: Use x in the model instead of purely random noise

        noise = self.generate_noise(x.size(0))
        out = self.model(noise)
        if self.fix_r is not None:
            if out.size(-1) < 3:
                out = torch.cat(
                    [out, self.fix_r * torch.ones_like(out[..., 0:1])], dim=-1
                )
            else:
                out[..., -1] = self.fix_r

        # Slice the output to the desired number of samples and dimensions
        out = out[:, : self.out_samples, :]

        if self.clip_output:

            clip_output_min, clip_output_max = self._get_clip_parameters()
            out = torch.clamp(out, clip_output_min, clip_output_max)

        return out


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
    # Energy-constrained Crystals Wasserstein GAN for the inverse design of crystal structures
    def __init__(
        self,
        input_channels,
        in_samples=2000,
        kernel_size=(1, 1),
        channels_coefficient=1,
        latent_dim=36,
    ):
        super().__init__()

        channels = [int(c*channels_coefficient) for c in [512, 512, 256]]

        # TODO: Make parameters configurable, need bigger model
        self.input_shape = (in_samples, input_channels)
        self.output_shape = (1,)

        self.mlflow_output_schema = Schema(
            [
                TensorSpec(shape=(-1, *self.output_shape), type=np.dtype(np.float32)),
            ]
        )
        self.mlflow_input_schema = Schema(
            [
                TensorSpec(shape=(-1, *self.input_shape), type=np.dtype(np.float32)),
            ]
        )

        assert (int(channels[2]*in_samples / latent_dim) == channels[2]*in_samples / latent_dim), "Latent dim must be a divisor of the input samples times the last channel"

        self.main = nn.Sequential(
            # Convolutional layer 2
            nn.Conv2d(
                input_channels,
                channels[0],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # # # Convolutional layer 2
            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # Convolutional layer 3
            nn.Conv2d(
                channels[1],
                channels[2],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.AdaptiveMaxPool2d(1),
            nn.Flatten(1, -1),
            nn.AvgPool1d(int(channels[2]*in_samples / latent_dim), count_include_pad=False), # Real paper has one per element, we have one element but want to simplify the data
            # nn.AdaptiveAvgPool2d(6), # Real paper has one per element, we have one element but want to simplify the data
        )

        self.fc_layers = nn.Sequential(
            # Fully connected layers
            # NOTE: Adjusted the width of the layers to the nearest power of 2 in comparison to the original paper
            nn.Linear(latent_dim, 1000),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(1000, 200),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(200, 10),
            # nn.Sigmoid(),
            nn.AdaptiveAvgPool1d(1),  # Average pooling, as in the CryinGAN paper
            # The other paper used fully connected layers all the way
        )

        for layer in self.main:  # TODO: check on this later
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("leaky_relu")
                )
                layer.bias.data.fill_(0.01)

            if isinstance(layer, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("leaky_relu")
                )
                layer.bias.data.fill_(0.01)

        for layer in self.fc_layers:
            # NOTE: Nice WETWET here, TODO: Fix this
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("leaky_relu")
                )
                layer.bias.data.fill_(0.01)

            if isinstance(layer, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("leaky_relu")
                )
                layer.bias.data.fill_(0.01)

    def forward(self, input):
        # First convolve each atom individually
        x = input.unsqueeze(-1).transpose(1, 2)
        # Then each atom is concatenated to the next one
        conv_output = self.main(x)
        # Finally flatten and pass through fully connected layers
        output = self.fc_layers(conv_output)

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
        channels_coefficient=1,
        clip_output: tuple = False,
        fix_r=None,
    ):

        # Inspired by: https://dl-acm-org.libproxy.aalto.fi/doi/abs/10.1145/3532213.3532218
        # 'Energy-constrained Crystals Wasserstein GAN for the inverse design of crystal structures',
        # @inproceedings{10.1145/3532213.3532218, author = {Hu, Peiyao and Ge, Binjing and Liu, Yirong and Huang, Wei}, title = {Energy-constrained Crystals Wasserstein GAN for the inverse design of crystal structures}, year = {2022}, isbn = {9781450396110}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, url = {https://doi-org.libproxy.aalto.fi/10.1145/3532213.3532218}, doi = {10.1145/3532213.3532218}, booktitle = {Proceedings of the 8th International Conference on Computing and Artificial Intelligence}, pages = {24–31}, numpages = {8}, keywords = {machine learning, generative model, Materials, Crystal structure, : Crystals}, location = {Tianjin, China}, series = {ICCAI '22} }

        super().__init__()

        self.clip_output = clip_output
        self.rand_features = rand_features
        self.channels_coefficient = channels_coefficient
        self.out_dimensions = out_dimensions
        self.out_samples = out_samples
        self.latent_dim = latent_dim
        self.latent_features = out_samples * latent_dim  # Samples x latent_dim
        self.input_shape = (out_samples, latent_dim)
        self.output_shape = (out_samples, out_dimensions)

        self.mlflow_output_schema = Schema(
            [
                TensorSpec(shape=(-1, -1), type=np.dtype(np.float32)),
            ]
        )
        self.mlflow_input_schema = Schema(
            [
                TensorSpec(shape=(-1, *self.input_shape), type=np.dtype(np.float32)),
            ]
        )

        self.fix_r = fix_r

        self.model = self._create_model(
            rand_features,
            latent_dim,
            out_samples,
            out_dimensions,
            channels_coefficient,
            self.latent_features,
        )

        for layer in self.model:  # TODO: check on this later
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("relu")
                )
                layer.bias.data.fill_(0.01)

            if isinstance(layer, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("relu")
                )
                layer.bias.data.fill_(0.01)

    @staticmethod
    def _create_model(
        rand_features,
        latent_dim,
        out_samples,
        out_dimensions,
        channels_coefficient,
        latent_features,
    ):

        channels=[int(c*channels_coefficient) for c in [256,512,256]]
        

        model = nn.Sequential(
            ## First fully connected layer
            # nn.Linear(in_features, 128 * 264), # 33k
            # nn.Linear(
            #     in_features=rand_features, out_features=rand_features
            # ),  # 12k, half size
            # nn.ReLU(True),
            nn.Linear(
                in_features=rand_features, out_features=latent_features
            ),  # 12k, half size
            nn.ReLU(),
            # # Reshape to (128, 264, 1, 1)
            # nn.Unflatten(1, (128, 264, 1)),
            nn.Unflatten(1, (latent_dim, out_samples, 1)),
            # # Transposed Convolution 1
            nn.ConvTranspose2d(
                latent_dim,
                channels[0],
                kernel_size=(1, out_dimensions),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(channels[0], eps=0.8),
            nn.ReLU(inplace=True),
            # # # Transposed Convolution 2
            nn.ConvTranspose2d(
                channels[0],
                channels[1],
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            # # Transposed Convolution 3
            nn.ConvTranspose2d(
                channels[1],
                channels[2],
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            # # Transposed Convolution 4 to get to 3 channels, 2000 samples and 1 feature
            nn.ConvTranspose2d(
                channels[2],
                1,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.Flatten(1, 2),
            nn.Sigmoid(),
        )
        return model

    def _get_clip_parameters(self):
        clip_output_min = torch.as_tensor(
            self.clip_output[0], device=self.model[0].weight.device
        )
        clip_output_max = torch.as_tensor(
            self.clip_output[1], device=self.model[0].weight.device
        )
        return clip_output_min, clip_output_max

    def generate_noise(self, batch_size):
        return rand(
            (batch_size, self.rand_features),
            device=self.model[0].weight.device,
            requires_grad=False,
        )

    def forward(self, x):
        # TODO: Use x in the model instead of purely random noise

        noise = self.generate_noise(x.size(0))

        out = self.model(noise)

        if self.fix_r is not None and self.out_dimensions > 2:
            if (out.size(-1) < 3):
                out = torch.cat(
                    [out, self.fix_r * torch.ones_like(out[..., 0:1])], dim=-1
                )
            else:
                out[..., -1] = self.fix_r

        # Do a softmax over the last dimensions to set radius

        if self.clip_output:

            clip_output_min, clip_output_max = self._get_clip_parameters()
            out = torch.clamp(out, clip_output_min, clip_output_max)

        return out


class CCCGeneratorWithDiffusion(CCCGenerator):
    """
    Generator for the CCC model with diffusion. The model is the same as the CCCGenerator, but with an additional diffusion layer.
    NOTE: Not really diffusion, but a vaguely similar idea.
    """

    def __init__(self, *args, real_sample=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_sample = real_sample[:, :2].unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _create_model(
        rand_features,  # For compatibility with parent class
        latent_dim,  # For compatibility with parent class
        out_samples,  # For compatibility with parent class
        out_dimensions,  # For compatibility with parent class
        channels_coefficient,
        latent_features,  # For compatibility with parent class
    ):
        model = nn.Sequential(  #
            # Transposed Convolution 1
            nn.ConvTranspose2d(
                1,
                128 * (2 ** (channels_coefficient - 1)),
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(128 * (2 ** (channels_coefficient - 1))),
            nn.ReLU(True),
            # # # Transposed Convolution 2
            nn.ConvTranspose2d(
                128 * (2 ** (channels_coefficient - 1)),
                64 * (2 ** (channels_coefficient - 1)),
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(64 * (2 ** (channels_coefficient - 1))),
            nn.ReLU(True),
            # # Transposed Convolution 3
            nn.ConvTranspose2d(
                64 * (2 ** (channels_coefficient - 1)),
                32 * (2 ** (channels_coefficient - 1)),
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(32 * (2 ** (channels_coefficient - 1))),
            nn.ReLU(True),
            # # Final transposed convolution to get to 3 channels, 2000 samples and 1 feature
            nn.ConvTranspose2d(
                32 * (2 ** (channels_coefficient - 1)),
                1,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            # nn.Sigmoid(), # not in the papers
            nn.Flatten(1, 2),
        )
        return model

    def generate_noise(self, batch_size):
        noise = (
            randn(
                (batch_size, 1, self.out_samples, self.out_dimensions),
                device=self.model[0].weight.device,
                requires_grad=False,
            )
            / 100 # Reduce the noise a bit
        )
        return self.real_sample + noise  
    

class CCCPredictor(nn.Module):
    """A CryinGAN style predictor to classify the atoms for a sample containing the x-y coordinates.
    """

    def __init__(self):
        super().__init__()
    




def build_model(**kwargs):
    """Interface to build selected model with parameters passed as kwargs"""

    _class = kwargs.pop("class")

    # Dynamically fetch the right class from this module
    try:
        model = getattr(sys.modules[__name__], _class)(**kwargs)
    except AttributeError:
        raise ValueError(f"Model class {_class} not found in module {__name__}")
    
    return model