import mlflow.experiments
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import mlflow
from matplotlib import pyplot as plt
import lightning as L
import psutil
import numpy as np

import os


from hypergrad import (
    AdamHD,
)  # This is a custom optimizer, you can use torch.optim.Adam instead

from src.plotting import plot_pointcloud, plot_sample_figures
from src.utils import build_optimizer_fn_from_config, build_run_name
from src.utils import log_nested_dict
from src.models.losses import build_loss_fn


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


class GAN(nn.Module):
    def __init__(
        self,
        trainset,
        testset,
        descriptor_loss=False,
        generator_model: HSGenerator = None,
        discriminator_model: HSDiscriminator = None,
        **run_params,
    ):
        super().__init__()
        device = run_params["training"]["device"]
        batch_size = run_params["training"]["batch_size"]

        if generator_model is None:
            generator = HSGenerator(**run_params["generator"])
        else:
            generator = generator_model
        if discriminator_model is None:
            discriminator = HSDiscriminator(**run_params["discriminator"])
        else:
            discriminator = discriminator_model

        self.run_params = run_params
        self.trainset = trainset
        self.testset = testset
        self.device = device

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        self.d_criterion = build_loss_fn(**run_params["training"]["d_loss"])
        self.g_criterion = build_loss_fn(**run_params["training"]["g_loss"])

        self.d_optimizer = build_optimizer_fn_from_config(
            run_params["training"]["optimizer_d"]
        )(
            self.discriminator.parameters()
        )  # AdamHD(params=self.generator.parameters(), lr=0.0002)
        self.g_optimizer = build_optimizer_fn_from_config(
            run_params["training"]["optimizer_g"]
        )(
            self.generator.parameters()
        )  #

    def train_n_epochs(
        self, epochs, batch_size=None, experiment_name=None, run_name=None, comment=None
    ):
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")  # NOTE: Seems slow

        if batch_size is None:
            batch_size = self.run_params["training"]["batch_size"]

        if run_name is None:
            run_name = build_run_name()

        print(
            f'Starting run {run_name}...\nTime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        )

        if experiment_name is not None:

            if not mlflow.search_experiments(
                filter_string=f"name = '{experiment_name}'"
            ):
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = mlflow.get_experiment_by_name(
                    experiment_name
                ).experiment_id

            run_params = {"experiment_id": experiment_id, "run_name": run_name}
        else:
            run_params = {"run_name": run_name}

        dataloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        if comment is not None:
            run_params["description"] = comment
        run_params["log_system_metrics"] = run_params.get(
            "log_system_metrics", True
        )  # Log system metrics by default
        with mlflow.start_run(**run_params):
            mlflow.log_params(self.run_params)
            log_nested_dict(self.run_params)

            # Log model size
            mlflow.log_metric(
                "g_model_size", sum(p.numel() for p in self.generator.parameters())
            )
            mlflow.log_metric(
                "d_model_size", sum(p.numel() for p in self.discriminator.parameters())
            )

            try:
                # Early stopping parameters
                mean_loss_d_prev = torch.inf
                mean_loss_g_prev = torch.inf

                patience = self.run_params["training"]["early_stopping_patience"]
                headstart = self.run_params["training"]["early_stopping_headstart"]
                # Log the naive scenario of no training
                fig = plot_sample_figures(
                    self.generator,
                    self.discriminator,
                    self.testset,
                    n=0,
                    plot_radius=True,
                    return_fig=True,
                )
                # Log pyplot figure to mlflow

                mlflow.log_figure(fig, f"generator_samples_epoch_pre.png")
                plt.close(fig)

                for epoch in tqdm(range(epochs)):
                    mean_loss_d, mean_loss_g = self._train_epoch(
                        epoch, batch_size=batch_size, dataloader=dataloader
                    )

                    # Early stopping if loss not decreasing
                    tolerance = 0.0001  # TODO: Move to config
                    if (
                        mean_loss_d > (mean_loss_d_prev - tolerance)
                        and mean_loss_g > (mean_loss_g_prev - tolerance)
                        and epoch > headstart  # NOTE: We don't want to stop too early
                    ):
                        patience -= 1
                        if patience == 0:
                            print("Early stopping")
                            break
                    else:
                        mean_loss_d_prev = mean_loss_d
                        mean_loss_g_prev = mean_loss_g
                        # model improving, reset patience
                        patience = self.run_params["training"][
                            "early_stopping_patience"
                        ]

            except KeyboardInterrupt:  # For jupyter notebook
                print("Interrupted")

    def _train_epoch(self, epoch, batch_size=None, dataloader=None, save_model=True):

        if batch_size is None:
            batch_size = self.run_params["training"]["batch_size"]

        if dataloader is None:
            dataloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.generator.train()
        self.discriminator.train()

        mean_loss_d = 0
        mean_loss_g = 0

        g_grad_norm = torch.tensor([0])

        for i, (descriptors, real_images) in enumerate(dataloader):
            real_images = real_images.to(self.device)
            descriptors = descriptors.to(self.device)

            real_labels = (
                torch.ones(real_images.size(0), 1).to(self.device) - 0.1
            )  # NOTE: A hack from 'Synthesising realistic 2D microstructures of unidirectional fibre-reinforced composites with a generative adversarial network'
            fake_labels = torch.zeros(real_images.size(0), 1).to(
                self.device
            )  # NOTE: The hack could be used in reverse

            # Train the discriminator
            self.generator.eval()
            self.d_optimizer.zero_grad()

            real_outputs = self.discriminator(real_images)
            d_loss_real = self.d_criterion(real_outputs, real_labels)
            d_loss_real.backward()

            fake_images = self.generator(
                torch.rand(*descriptors.size()).to(self.device)
            ).detach()
            # NOTE: Should the fake images be detached to avoid backpropagating through the generator?
            fake_outputs = self.discriminator(fake_images)
            d_loss_fake = self.d_criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake

            d_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), 50, error_if_nonfinite=True
            )  # Clip gradients

            # Give the generator a headstart
            if self.run_params["training"]["generator_headstart"] < epoch:
                self.d_optimizer.step()

            mean_loss_d += d_loss.item()

            # Train the generator
            if i % self.run_params["training"]["training_ratio_dg"] != 0:
                continue
                # Only update the generator every n steps

            self.generator.train()  # NOTE: This is not in the original code
            self.discriminator.eval()  # NOTE: This is not in the original code
            self.g_optimizer.zero_grad()

            fake_images = self.generator(
                torch.rand(*descriptors.size()).to(self.device)
            )
            fake_outputs = self.discriminator(fake_images)

            g_loss = self.g_criterion(
                real_images, fake_images, fake_outputs, real_labels
            )  # We want the generator to generate images that the discriminator thinks are real
            g_loss.backward()

            g_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 50, error_if_nonfinite=True
            )  # Clip gradients
            self.g_optimizer.step()

            mean_loss_g += g_loss.item()

        # Log the optimizer hyperparameters
        if hasattr(self.d_optimizer, "param_groups"):
            lr_d = self.d_optimizer.param_groups[0]["lr"]
            lr_g = self.g_optimizer.param_groups[0]["lr"]

            mlflow.log_metric("lr_d", lr_d, step=epoch)
            mlflow.log_metric("lr_g", lr_g, step=epoch)

        if (
            epoch % self.run_params["training"]["log_image_frequency"]
        ) == 0:  # TODO: Fix this

            fig = plot_sample_figures(
                self.generator,
                self.discriminator,
                self.testset,
                n=int(epoch % 4),  # NOTE: This is a hack
                plot_radius=True,
                return_fig=True,
            )
            # Log pyplot figure to mlflow

            epoch_str = str(epoch).zfill(4)

            mlflow.log_figure(fig, f"generator_samples_epoch_{epoch_str}.png")
            plt.close(fig)

        if save_model:
            pass
            # Log the model
            # sklearn.log_model(
            #     sk_model=lr,
            #     artifact_path="iris_model",
            #     signature=signature,
            #     input_example=X_train,
            #     registered_model_name="tracking-quickstart",
            # )

        # print(
        #     f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, Epoch {epoch}, D loss: {d_loss.item()}, G_GAN loss: {g_loss_gan.item()}, G_P loss: {g_loss_physical.item()}'
        # )

        mean_loss_d /= len(self.trainset)
        mean_loss_g /= len(self.trainset)

        # Log metrics with mlflow
        g_loss_gan = self.g_criterion.prev_gan_loss.item()
        g_loss_radius = self.g_criterion.prev_radius_loss.item()
        g_loss_density = self.g_criterion.prev_grid_density_loss.item()
        g_physical_feasibility_loss = (
            self.g_criterion.prev_physical_feasibility_loss.item()
        )

        mlflow.log_metric("D_loss", mean_loss_d, step=epoch)
        mlflow.log_metric("G_loss", mean_loss_g, step=epoch)
        mlflow.log_metric("G_Density_loss", g_loss_density, step=epoch)
        mlflow.log_metric("G_R_loss", g_loss_radius, step=epoch)
        mlflow.log_metric("G_GAN_loss", g_loss_gan, step=epoch)
        mlflow.log_metric("G_Feasibility_loss", g_physical_feasibility_loss, step=epoch)

        # Log gradients with mlflow

        mlflow.log_metric("D_grad_norm", d_grad_norm, step=epoch)
        mlflow.log_metric("G_grad_norm", g_grad_norm, step=epoch)

        # Log system metrics # NOTE: These are logged elsewhere
        # mlflow.log_metric("CPU", psutil.cpu_percent(), step=epoch)
        # mlflow.log_metric("RAM", psutil.virtual_memory().percent, step=epoch)

        return mean_loss_d, mean_loss_g

    def generate(self, input):
        self.generator.eval()
        return self.generator(input.to(self.device)).detach().cpu()
