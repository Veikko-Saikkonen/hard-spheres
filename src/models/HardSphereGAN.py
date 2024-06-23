import mlflow.experiments
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import mlflow
from matplotlib import pyplot as plt
import lightning as L

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


from hypergrad import (
    AdamHD,
)  # This is a custom optimizer, you can use torch.optim.Adam instead

from src.plotting import plot_pointcloud, plot_sample_figures
from src.utils import build_optimizer_fn_from_config, build_run_name


class HSGenerator(nn.Module):
    def __init__(self, in_dim, latent_dim=27, output_max_samples=2000, cnn_channels=16):
        super().__init__()
        # Takes in the input descriptors and returns the output point cloud

        assert latent_dim % 3 == 0, "Latent dim needs to be multiples of 3"

        # TODO: Move these to config
        # Maps descriptors to latent space
        in_dim = 1  # We have a single descriptor

        kernel_y = 64
        cnn_layers = 3

        latent_max_samples = (
            output_max_samples + cnn_layers * kernel_y
        )  # We will add some padding to the latent space
        kernel_x = latent_dim // cnn_layers

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

        self.latent2pointcloud = nn.Sequential(
            nn.Conv2d(
                1,
                cnn_channels,
                kernel_size=(kernel_y + 1, kernel_x),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                cnn_channels,
                cnn_channels,
                kernel_size=(kernel_y + 1, kernel_x),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                cnn_channels,
                1,
                kernel_size=(kernel_y + 1, kernel_x),
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.output_max_samples = output_max_samples
        self.latent_max_samples = latent_max_samples
        self.latent_dim = latent_dim

    def get_z(self, batch_size):
        z = torch.rand(
            batch_size,
            self.latent_max_samples,
            self.latent_dim,
            device=self.desc2latent[0].weight.device,
        )

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
        return torch.clip(x, 0, 1)


class HSGenerator(nn.Module):
    def __init__(self, in_dim, latent_dim=27, output_max_samples=2000, cnn_channels=16):
        super().__init__()
        # Takes in the input descriptors and returns the output point cloud

        assert latent_dim % 3 == 0, "Latent dim needs to be multiples of 3"

        # TODO: Move these to config
        # Maps descriptors to latent space
        in_dim = 1  # We have a single descriptor

        kernel_y = 64
        cnn_layers = 3

        latent_max_samples = (
            output_max_samples + cnn_layers * kernel_y
        )  # We will add some padding to the latent space
        kernel_x = latent_dim // cnn_layers

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

        self.latent2pointcloud = nn.Sequential(
            nn.Conv2d(
                1,
                cnn_channels,
                kernel_size=(kernel_y + 1, kernel_x),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                cnn_channels,
                cnn_channels,
                kernel_size=(kernel_y + 1, kernel_x),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                cnn_channels,
                1,
                kernel_size=(kernel_y + 1, kernel_x),
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.output_max_samples = output_max_samples
        self.latent_max_samples = latent_max_samples
        self.latent_dim = latent_dim

    def get_z(self, batch_size):
        return torch.rand(
            batch_size,
            self.latent_max_samples,
            self.latent_dim,
            device=self.desc2latent[0].weight.device,
        )

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
        super(GAN, self).__init__()
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
        self.criterion = nn.BCELoss()  # TODO: Think of this

        self.d_optimizer = build_optimizer_fn_from_config(
            run_params["training"]["optimizer_d"]
        )(
            self.discriminator.parameters()
        )  # AdamHD(params=self.generator.parameters(), lr=0.0002)
        self.g_optimizer = build_optimizer_fn_from_config(
            run_params["training"]["optimizer_g"]
        )(
            self.generator.parameters()
        )  # AdamHD(params=self.generator.parameters(), lr=0.0002)

        if descriptor_loss:
            self.descriptor_criterion = nn.MSELoss()
            # TODO: Implement

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

        with mlflow.start_run(**run_params):
            mlflow.log_params(self.run_params)
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

    def _feasibility_loss(self, fake_images):
        # Loss based on the feasibility of the point cloud
        # NOTE: This is not used for now
        return 0

    def _physical_loss(self, real_images, fake_images):
        # Loss based on the physical properties of the point cloud

        # Loss based on sum of radiuses
        real_radius = real_images[:, 2].mean()
        fake_radius = fake_images[:, 2].mean()
        radius_loss = torch.abs(real_radius - fake_radius)

        # Loss based on the variance of radii
        real_r_var = real_images[:, 2].var()
        fake_r_var = fake_images[:, 2].var()
        r_var_loss = torch.abs(real_r_var - fake_r_var)

        # Loss based on mean and variance of x and y # NOTE: May not be meaningful, maybe use Min, max etc. or don't use
        # real_x_mean = real_images[:, 0].mean()
        # fake_x_mean = fake_images[:, 0].mean()
        # x_mean_loss = torch.abs(real_x_mean - fake_x_mean)

        # real_x_var = real_images[:, 0].var()
        # fake_x_var = fake_images[:, 0].var()
        # x_var_loss = torch.abs(real_x_var - fake_x_var)

        # real_y_mean = real_images[:, 1].mean()
        # fake_y_mean = fake_images[:, 1].mean()
        # y_mean_loss = torch.abs(real_y_mean - fake_y_mean)

        # real_y_var = real_images[:, 1].var()
        # fake_y_var = fake_images[:, 1].var()
        # y_var_loss = torch.abs(real_y_var - fake_y_var)

        # TODO Add more physical properties, phi etc.
        loss = (
            radius_loss
            + r_var_loss
            # + x_mean_loss
            # + x_var_loss
            # + y_mean_loss
            # + y_var_loss
        ) / 2
        return loss

    def _train_epoch(self, epoch, batch_size=None, dataloader=None, save_model=True):

        if batch_size is None:
            batch_size = self.run_params["training"]["batch_size"]

        if dataloader is None:
            dataloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.generator.train()
        self.discriminator.train()

        mean_loss_d = 0
        mean_loss_g = 0

        for descriptors, real_images in dataloader:
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
            d_loss_real = self.criterion(real_outputs, real_labels)
            d_loss_real.backward()

            fake_images = self.generator(descriptors).detach()
            # NOTE: Should the fake images be detached to avoid backpropagating through the generator?
            fake_outputs = self.discriminator(fake_images)
            d_loss_fake = self.criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake

            d_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), 50, error_if_nonfinite=True
            )  # Clip gradients
            self.d_optimizer.step()
            mean_loss_d += d_loss.item()

            # Train the generator
            self.generator.train()  # NOTE: This is not in the original code
            self.discriminator.eval()  # NOTE: This is not in the original code
            self.g_optimizer.zero_grad()

            fake_images = self.generator(descriptors)
            fake_outputs = self.discriminator(fake_images)
            g_loss_gan = self.criterion(
                fake_outputs, real_labels
            )  # We want the generator to generate images that the discriminator thinks are real
            g_loss_physical = self._physical_loss(real_images, fake_images)

            g_loss = (
                g_loss_gan + g_loss_physical
            )  # NOTE: The physical loss is not used for now
            g_loss.backward()

            g_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 50, error_if_nonfinite=True
            )  # Clip gradients
            self.g_optimizer.step()

            mean_loss_g += g_loss.item()

        # Log the optimizer hyperparameters
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
            mlflow.log_figure(fig, f"generator_samples_epoch_{epoch}.png")
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
        mlflow.log_metric("D_loss", mean_loss_d, step=epoch)
        mlflow.log_metric("G_loss", mean_loss_g, step=epoch)
        mlflow.log_metric("G_P_loss", g_loss_physical.item(), step=epoch)
        mlflow.log_metric("G_GAN_loss", g_loss_gan.item(), step=epoch)

        # Log gradients with mlflow

        mlflow.log_metric("D_grad_norm", d_grad_norm, step=epoch)
        mlflow.log_metric("G_grad_norm", g_grad_norm, step=epoch)

        return mean_loss_d, mean_loss_g

    def generate(self, input):
        self.generator.eval()
        return self.generator(input.to(self.device)).detach().cpu()


# Add a PyTorch Lighting compatible module


class HardSphereGAN(L.LightningModule):
    def __init__(
        self,
        trainset,
        testset,
        descriptor_loss=False,
        **run_params,
    ):
        super(HardSphereGAN, self).__init__()
        device = run_params["training"]["device"]
        batch_size = run_params["training"]["batch_size"]

        generator = HSGenerator(**run_params["generator"])
        discriminator = HSDiscriminator(**run_params["discriminator"])

        self.run_params = run_params
        self.trainset = trainset
        self.testset = testset
        self.device = device

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.criterion = nn.BCELoss()  # TODO: Think of this

        if descriptor_loss:
            self.descriptor_criterion = nn.MSELoss()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        descriptors, real_images = batch
        real_images = real_images.to(self.device)
        descriptors = descriptors.to(self.device)

        real_labels = torch.ones(real_images.size(0), 1).to(self.device) - 0.1
        fake_labels = torch.zeros(real_images.size(0), 1).to(self.device)

        if optimizer_idx == 0:
            self.d_optimizer.zero_grad()

            real_outputs = self.discriminator(real_images)
            d_loss_real = self.criterion(real_outputs, real_labels)
            d_loss_real.backward()

            fake_images = self.generator(descriptors).detach()
            fake_outputs = self.discriminator(fake_images)
            d_loss_fake = self.criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake

            d_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), 50, error_if_nonfinite=True
            )
            self.d_optimizer.step()

            return d_loss

        if optimizer_idx == 1:
            self.g_optimizer.zero_grad()

            fake_images = self.generator(descriptors)
            fake_outputs = self.discriminator(fake_images)
            g_loss_gan = self.criterion(fake_outputs, real_labels)
            g_loss_physical = self._physical_loss(real_images, fake_images)

            g_loss = g_loss_gan + g_loss_physical
            g_loss.backward()

            g_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 50, error_if_nonfinite=True
            )
            self.g_optimizer.step()

            return g_loss

    def _physical_loss(self, real_images, fake_images):
        real_radius = real_images[:, 2].mean()
        fake_radius = fake_images[:, 2].mean()
        radius_loss = torch.abs(real_radius - fake_radius)

        real_r_var = real_images[:, 2].var()
        fake_r_var = fake_images[:, 2].var()
        r_var_loss = torch.abs(real_r_var - fake_r_var)

        loss = (radius_loss + r_var_loss) / 2
        return loss

    def configure_optimizers(self):

        self.d_optimizer = build_optimizer_fn_from_config(
            self.run_params["training"]["optimizer_d"]
        )(self.discriminator.parameters())
        self.g_optimizer = build_optimizer_fn_from_config(
            self.run_params["training"]["optimizer_g"]
        )(self.generator.parameters())

        return [self.d_optimizer, self.g_optimizer], []

    def generate(self, input):
        self.generator.eval()
        return self.generator(input.to(self.device)).detach().cpu()

    def training_epoch_end(self, outputs):
        mean_loss_d = torch.stack([x for x in outputs[0]]).mean()
        mean_loss_g = torch.stack([x for x in outputs[1]]).mean()

        return mean_loss_d, mean_loss_g
