from torch import nn
import torch


class HSGenerator(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim=27,
        output_max_samples=2000,
        cnn_channels=2,
        kernel_y=128,
        kernel_x=5,
        latent2pointcloud_layers=3,
    ):
        super().__init__()
        # Takes in the input descriptors and returns the output point cloud

        # Maps descriptors to latent space
        self.activation_fn = nn.Sigmoid

        in_dim = 1  # We have a single descriptor

        latent_max_samples = output_max_samples  # The maximum number of samples in the latent space is the same as the output space
        # Takes in the input descriptors and returns the output point cloud

        self.desc2latent = nn.Sequential(
            # Input to latent space
            nn.ConvTranspose1d(
                in_dim,
                latent_dim // 2,
                kernel_size=latent_max_samples // 2,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ConvTranspose1d(
                latent_dim // 2,
                latent_dim,
                kernel_size=latent_max_samples // 2 - 1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.latent2pointcloud = self._make_latent_to_pointcloud(
            n_layers=latent2pointcloud_layers,
            kernel_y=kernel_y,
            kernel_x=kernel_x,
            cnn_channels=cnn_channels,
        )

        self.bn = nn.BatchNorm1d(3)  # Output normalization

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.output_max_samples = output_max_samples
        self.latent_max_samples = latent_max_samples
        self.latent_dim = latent_dim

    def _make_latent_to_pointcloud(self, n_layers, kernel_y, kernel_x, cnn_channels):
        _input = nn.Conv2d(
            1,
            cnn_channels,
            kernel_size=(kernel_y + 1, kernel_x),
            stride=1,
            padding="same",
            bias=True,
        )
        _layers = [_input]

        for i in range(n_layers):
            _layers.append(self.activation_fn())
            _layers.append(
                nn.Conv2d(
                    cnn_channels,
                    cnn_channels,
                    kernel_size=(kernel_y + 1, kernel_x),
                    stride=1,
                    padding="same",
                    bias=False,
                )
            )
        _layers.append(
            nn.Conv2d(
                cnn_channels,
                out_channels=1,
                kernel_size=(kernel_y + 1, kernel_x),
                stride=1,
                padding="same",
                bias=True,
            )
        )

        model = nn.Sequential(*_layers)

        for layer in model:
            if hasattr(layer, "weight"):
                # nn.init.uniform_(layer.weight, a=-0.01, b=0.01)
                nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("sigmoid")
                )
            # if hasattr(layer, "bias"):
            #     nn.init.uniform_(layer.bias, a=-0.01, b=0.01)

        return model

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            self.activation_fn(),
        )

    def get_z(self, batch_size):
        # z = torch.rand(
        #     batch_size,
        #     self.latent_max_samples,
        #     self.latent_dim,
        #     device=self.desc2latent[0].weight.device,
        # )
        device = device = self.desc2latent[0].weight.device

        # Set seed
        torch.manual_seed(0)  # NOTE: This is a hack

        zx = torch.rand(
            batch_size, self.latent_max_samples, 1, device=device
        )  # X is uniform
        zy = torch.rand(
            batch_size, self.latent_max_samples, 1, device=device
        )  # Y is uniform
        # R is inverse exponential
        zr = torch.tensor(
            np.random.exponential(0.3, (batch_size, self.latent_max_samples, 1)),
            dtype=torch.float32,
            device=device,
        )
        z = torch.cat([zx, zy, zr], dim=-1)

        # Sort the z values the same way the labels are sorted
        # Sort the z values the same way the labels are sorted
        sort_z = torch.sqrt(((z[:, :, 0] - 0.5) ** 2) + ((z[:, :, 1] - 0.5) ** 2))
        sort_z, indices = torch.sort(sort_z, dim=1)
        z = z[torch.arange(z.size(0)).unsqueeze(1), indices]
        return z

    def forward(self, d: torch.tensor):

        # Input d: Batch x Descriptors x 1 # TODO: In the future a single descriptor will be a 1D tensor, changing dimensions to Batch x Descriptors x NFeatures
        # Output: Batch x Samples x 4 (class, x, y, r)
        batch_size = d.shape[0]

        z = self.get_z(batch_size)

        x = d
        # x = (self.desc2latent(x).transpose(-1, -2) + z) / 2
        # TODO: Add the descriptors
        x = z
        x = self.latent2pointcloud(x.unsqueeze(1)).squeeze(1)

        # Batch normalization

        # x = x.transpose(-1, -2)
        # x = self.bn(x).transpose(-1, -2)

        return torch.clip(x, 0, 1)


class HSDiscriminator(nn.Module):
    def __init__(self, channels_img=4, features_d=12):
        # Discriminator takes in the point cloud and returns a list of predicted labels (real/fake)
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d // 2, kernel_size=(64, 2), stride=(2, 1)
            ),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(
                features_d // 2, features_d, kernel_size=(32, 2), stride=(3, 1)
            ),
            self._block(features_d, features_d, kernel_size=(16, 2), stride=(1, 1)),
            nn.Conv2d(features_d, 1, kernel_size=(4, 2)),  # 1x1
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.MaxPool1d(5),
            nn.Linear(178, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x.unsqueeze(1))
