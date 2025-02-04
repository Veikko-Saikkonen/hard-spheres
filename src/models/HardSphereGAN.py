import mlflow.experiments
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import mlflow
from matplotlib import pyplot as plt
from pathlib import Path

from src.plotting import plot_pointcloud, plot_sample_figures, plot_sample_distributions
from src.utils import build_optimizer_fn_from_config, build_run_name
from src.utils import log_nested_dict
from src.models.losses import build_loss_fn
from src.models.losses import CryinGANDiscriminatorLoss
from src.models import CryinGAN
from src.metrics import packing_fraction


class GAN(nn.Module):
    def __init__(
        self,
        trainset,
        testset,
        generator_model: torch.nn.Module = None,
        discriminator_model: torch.nn.Module = None,
        predictor_model: torch.nn.Module = None,
        **run_params,
    ):
        super().__init__()
        device = run_params["training"]["device"]
        batch_size = run_params["training"]["batch_size"]

        if generator_model is None:
            # This means the user has not provided a generator model, but has provided parameters for the generator
            generator_params = run_params["generator"]
            self.generator = CryinGAN.build_model(**run_params["generator"])
        else:
            self.generator = generator_model
            run_params["generator"] = {}
            run_params["generator"]["name"] = type(self.generator).__name__


        if discriminator_model is None:
            self.discriminator = CryinGAN.build_model(**run_params["discriminator"])
        else:
            self.discriminator = discriminator_model
            run_params["discriminator"] = {}
            run_params["discriminator"]["name"] = type(self.discriminator).__name__

        if predictor_model is None:
            pass # NOTE: Implement this
            # self.predictor = CryinGAN.build_model(**run_params["predictor"])
        else:
            self.predictor = predictor_model
            run_params["predictor"] = {}
            run_params["predictor"]["name"] = type(self.predictor).__name__

        self.run_params = run_params
        self.trainset = trainset
        self.testset = testset
        self.device = device

        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        # self.predictor = self.predictor.to(device)

        self.d_criterion = build_loss_fn(**run_params["training"]["d_loss"])
        self.g_criterion = build_loss_fn(**run_params["training"]["g_loss"])

        self.d_optimizer = build_optimizer_fn_from_config(
            run_params["training"]["optimizer_d"]
        )(self.discriminator.parameters())
        self.g_optimizer = build_optimizer_fn_from_config(
            run_params["training"]["optimizer_g"]
        )(self.generator.parameters())

        if self.trainset.y.shape[-1] == 3:
            self.plot_radius = True
        else:
            self.plot_radius = False

        # Register dataset with mlflow

    def train_n_epochs(
        self,
        epochs,
        batch_size=None,
        experiment_name=None,
        run_name=None,
        comment=None,
        save_model=True,
        requirements_file: Path = None,
    ):

        if requirements_file:
            requirements_file = requirements_file.resolve()
            requirements_file = str(requirements_file)
            print(f"Using requirements file: {requirements_file}")

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

        if comment is not None:
            run_params["description"] = comment
        run_params["log_system_metrics"] = run_params.get(
            "log_system_metrics", True
        )  # Log system metrics by default
        with mlflow.start_run(**run_params):
            # Log dataset
            ml_data_train = mlflow.data.from_numpy(
                features=self.trainset.x.detach().cpu().numpy(),
                targets=self.trainset.y.detach().cpu().numpy(),
                name="trainset",
            )
            ml_data_test = mlflow.data.from_numpy(
                features=self.testset.x.detach().cpu().numpy(),
                targets=self.testset.y.detach().cpu().numpy(),
                name="testset",
            )
            mlflow.log_input(ml_data_train, context="trainset")
            mlflow.log_input(ml_data_test, context="testset")

            # Create dataloader
            dataloader = DataLoader(
                self.trainset, batch_size=batch_size, shuffle=True, drop_last=False
            )

            # Log run parameters
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
                g_grad_norm_prev = torch.inf
                d_grad_norm_prev = torch.inf

                patience = self.run_params["training"]["early_stopping_patience"]
                headstart = self.run_params["training"]["early_stopping_headstart"]
                early_stopping_tolerance = self.run_params["training"][
                    "early_stopping_tolerance"
                ]
                # Log the naive scenario of no training
                fig = plot_sample_figures(
                    self.generator,
                    self.discriminator,
                    self.testset,
                    n=0,
                    plot_radius=self.plot_radius,
                    return_fig=True,
                )
                # Log pyplot figure to mlflow

                mlflow.log_figure(fig, f"generator_samples_epoch_pre.png")
                plt.close(fig)

                for epoch in tqdm(range(epochs)):
                    mean_loss_d, mean_loss_g, g_grad_norm, d_grad_norm = (
                        self._train_epoch(
                            epoch, batch_size=batch_size, dataloader=dataloader
                        )
                    )

                    # Early stopping if loss not decreasing
                    if (
                        # Look at gradients
                        abs(g_grad_norm - g_grad_norm_prev)  # Change of gradient norm
                        < early_stopping_tolerance
                    ):
                        patience -= 1
                        if patience == 0:
                            print("Early stopping")
                            break
                    else:
                        mean_loss_d_prev = mean_loss_d
                        mean_loss_g_prev = mean_loss_g
                        g_grad_norm_prev = g_grad_norm
                        d_grad_norm_prev = d_grad_norm
                        # model improving, reset patience
                        patience = self.run_params["training"][
                            "early_stopping_patience"
                        ]

            except KeyboardInterrupt:  # For jupyter notebook
                print("Interrupted")

            if save_model:
                # Save the trained model to MLflow.
                generator_signature = mlflow.models.ModelSignature(
                    inputs=self.generator.mlflow_input_schema,
                    outputs=self.generator.mlflow_output_schema,
                )

                discriminator_signature = mlflow.models.ModelSignature(
                    inputs=self.discriminator.mlflow_input_schema,
                    outputs=self.discriminator.mlflow_output_schema,
                )

                print("Logging models to mlflow")
                mlflow.pytorch.log_model(
                    self.generator,
                    "generator",
                    signature=generator_signature,
                    pip_requirements=requirements_file,
                )
                mlflow.pytorch.log_model(
                    self.discriminator,
                    "discriminator",
                    signature=discriminator_signature,
                    pip_requirements=requirements_file,
                )

    def _train_epoch(self, epoch, batch_size=None, dataloader=None):

        if batch_size is None:
            batch_size = self.run_params["training"]["batch_size"]

        if dataloader is None:
            dataloader = DataLoader(
                self.trainset, batch_size=batch_size, shuffle=True, drop_last=False
            )

        self.generator.train()
        self.discriminator.train()

        mean_loss_d = 0
        mean_loss_g = 0

        g_grad_norm = torch.tensor([0])

        for i, (descriptors, real_images) in enumerate(dataloader):
            real_images = real_images.to(self.device)
            descriptors = descriptors.to(self.device)

            # Train the discriminator
            self.generator.eval()
            self.discriminator.train()
            self.d_optimizer.zero_grad()

            real_outputs = self.discriminator(real_images)

            fake_images = self.generator(
                torch.rand(*descriptors.size()).to(self.device)
            ).detach()
            # NOTE: Should the fake images be detached to avoid backpropagating through the generator? We dont want to update the generator weights here
            fake_outputs = self.discriminator(fake_images)

            if (
                self.run_params["training"]["d_loss"]["name"]
                == "CryinGANDiscriminatorLoss"
            ):
                d_loss = self.d_criterion(
                    real_outputs,
                    fake_outputs,
                    real_images,
                    fake_images,
                    self.discriminator,
                )
            else:
                # Wasserstein GAN
                real_labels = torch.ones_like(real_outputs) - 0.1
                fake_labels = torch.zeros_like(fake_outputs) + 0.1

                d_loss = self.d_criterion(real_outputs, real_labels) + self.d_criterion(
                    fake_outputs, fake_labels
                )

            d_loss.backward()

            d_grad_clip_limit = 50
            d_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                d_grad_clip_limit,
                error_if_nonfinite=True,
            )  # Clip gradients
            d_grad_norm = torch.min(
                d_grad_norm,
                torch.tensor([d_grad_clip_limit], device=d_grad_norm.device),
            )  # Avoid reporting infinity

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
                real_images, fake_images, fake_outputs
            )  # We want the generator to generate images that the discriminator thinks are real
            g_loss.backward()

            g_grad_clip_limit = 50
            g_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), g_grad_clip_limit, error_if_nonfinite=True
            )  # Clip gradients, TODO: move to config
            g_grad_norm = torch.min(
                g_grad_norm,
                torch.tensor([g_grad_clip_limit], device=g_grad_norm.device),
            )
            self.g_optimizer.step()

            mean_loss_g += g_loss.item()

            # Log metrics

            if self.run_params["metrics"]["packing_fraction"]:
                packing_fraction_real = packing_fraction(real_images, self.run_params["metrics"]["packing_fraction_fix_r"], self.run_params["metrics"]["packing_fraction_box_size"])
                packing_fraction_fake = packing_fraction(fake_images, self.run_params["metrics"]["packing_fraction_fix_r"], self.run_params["metrics"]["packing_fraction_box_size"])

                mlflow.log_metric("packing_fraction_real", packing_fraction_real, step=epoch)
                mlflow.log_metric("packing_fraction_fake", packing_fraction_fake, step=epoch)

            # output example s
            # discriminator_output = self.discriminator(fake_images)
            # discriminator_output_real = self.discriminator(real_images)

            # # Transfer to csv
            # import pandas as pd

            # print(real_images.shape)
            # pd.Series(
            #     discriminator_output_real.to("cpu").detach().numpy().flatten()
            # ).to_csv("discriminator_output_real.csv")
            # pd.Series(discriminator_output.to("cpu").detach().numpy().flatten()).to_csv(
            #     "discriminator_output.csv"
            # )

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
                self.testset,  # NOTE: Should we visualize train set too?
                n=torch.randint(
                    0, len(self.testset) - 1, (1,)
                ).item(),  # Plot a random sample
                plot_radius=self.plot_radius,
                return_fig=True,
            )
            # Log pyplot figure to mlflow

            epoch_str = str(epoch).zfill(4)

            mlflow.log_figure(fig, f"generator_samples_epoch_{epoch_str}.png")
            plt.close(fig)

            fig = plot_sample_distributions(
                self.generator,
                dataset=self.testset,
                n=int(epoch % 4),
                return_fig=True,
                plot_radius=self.plot_radius,
                plot_distances=True,
            )

            mlflow.log_figure(fig, f"generator_distributions_epoch_{epoch_str}.png")
            plt.close(fig)

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
        g_loss_distance = self.g_criterion.prev_distance_loss.item()
        g_grid_order_loss = self.g_criterion.prev_grid_order_loss.item()

        if "prev_gradient_penalty" in self.d_criterion.__dict__:
            d_penalty_gradient = self.d_criterion.prev_gradient_penalty.item()
            d_loss_gan = self.d_criterion.prev_gan_loss.item()
        else:
            d_penalty_gradient = 0
            d_loss_gan = 0

        mlflow.log_metric("D_loss", mean_loss_d, step=epoch)
        mlflow.log_metric("G_loss", mean_loss_g, step=epoch)
        if g_loss_density > 0:
            mlflow.log_metric("G_Density_loss", g_loss_density, step=epoch)
        if g_loss_radius > 0:
            mlflow.log_metric("G_Radius_loss", g_loss_radius, step=epoch)
        if g_loss_gan > 0:
            mlflow.log_metric("G_GAN_loss", g_loss_gan, step=epoch)
        if g_physical_feasibility_loss > 0:
            mlflow.log_metric(
                "G_Feasibility_loss", g_physical_feasibility_loss, step=epoch
            )
        if g_loss_distance > 0:
            mlflow.log_metric("G_Distance_loss", g_loss_distance, step=epoch)
        if g_grid_order_loss is not None:
            mlflow.log_metric("G_Grid_loss", g_grid_order_loss, step=epoch)
            

        mlflow.log_metric("D_Gradient_penalty", d_penalty_gradient, step=epoch)
        mlflow.log_metric("D_GAN_loss", d_loss_gan, step=epoch)

        # Log gradients with mlflow

        # d_grad_norm = torch.tensor([0])
        # g_grad_norm = torch.tensor([0])

        mlflow.log_metric("D_grad_norm", d_grad_norm, step=epoch)
        mlflow.log_metric("G_grad_norm", g_grad_norm, step=epoch)

        # Log system metrics # NOTE: These are logged elsewhere
        # mlflow.log_metric("CPU", psutil.cpu_percent(), step=epoch)
        # mlflow.log_metric("RAM", psutil.virtual_memory().percent, step=epoch)

        return mean_loss_d, mean_loss_g, g_grad_norm, d_grad_norm

    def generate(self, input):
        self.generator.eval()
        return self.generator(input.to(self.device)).detach().cpu()
