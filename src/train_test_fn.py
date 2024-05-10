import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch import autograd
import torchvision

import numpy as np
import pandas as pd
from contextlib import nullcontext
from copy import deepcopy


# Create a training method for the gan, containing logging results to MLFlow


def train_gan(
    model,
    train_loader,
    epochs,
    device,
    optimizer_gen,
    optimizer_disc,
    criterion,
    fixed_noise,
    mlflow,
    run_name,
):
    batch_size = train_loader.batch_size

    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(train_loader):
            real = real.to(device)
            noise = torch.randn((batch_size, model.z_dim, 1, 1)).to(device)
            fake = model.gen(noise)
            disc_real = model.disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = model.disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            model._reset_gradients()
            loss_disc.backward(retain_graph=True)
            optimizer_disc.step()

            output = model.disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            model._reset_gradients()
            loss_gen.backward()
            optimizer_gen.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} \
                          Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )
                with torch.no_grad():
                    fake = model.gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )
                    mlflow.log_image(img_grid_real, f"Real Epoch {epoch}")
                    mlflow.log_image(img_grid_fake, f"Fake Epoch {epoch}")
                    mlflow.log_metric("Loss D", loss_disc)
                    mlflow.log_metric("Loss G", loss_gen)
                    mlflow.log_metric("Epoch", epoch)
                    mlflow.set_tag("Run Name", run_name)

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f"saved_models/{run_name}_epoch_{epoch}.pt",
            )
            print(f"Saving model at epoch {epoch}")

    torch.save(model.state_dict(), f"saved_models/{run_name}_final.pt")
    print("Saving final model")
    return model
