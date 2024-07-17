from matplotlib import pyplot as plt


def plot_pointcloud(pointcloud, ax=None, plot_radius=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    xs = pointcloud[:, 0]
    ys = pointcloud[:, 1]

    if plot_radius:
        rs = pointcloud[:, 2] + 0.1  # Add 0.1 to the radius to make it visible

    if plot_radius:
        ax.scatter(xs, ys, c=rs, s=rs * 30, alpha=0.5)
    else:
        ax.scatter(
            xs, ys, alpha=0.5
        )  # TODO: Size of the points is not correct in comparison to the grid size

    return ax


def plot_sample_distributions(
    generator,
    dataset=None,
    sample_y=None,
    n=5,
    return_fig=False,
    plot_radius=True,
    plot_distances=False,
):
    sample_generated_y = generator(dataset[n][0]).detach().cpu()

    if sample_y is None:
        try:
            sample_y = dataset[n][1].unsqueeze(0)
        except TypeError:
            raise ValueError("Either 'sample_y' or 'dataset' must be defined!")
    sample_y = sample_y.cpu()

    n_figs = 2 + int(plot_radius) + int(plot_distances)

    fig, ax = plt.subplots(1, n_figs, figsize=(13, 5))

    xs_real = sample_y[:, :, 0]
    ys_real = sample_y[:, :, 1]

    xs_fake = sample_generated_y[:, :, 0]
    ys_fake = sample_generated_y[:, :, 1]

    # Plot the distributions of the different variables in different figures
    ax[0].hist(xs_real.numpy().flatten(), bins=20, alpha=0.5, label="Real X")
    ax[0].hist(xs_fake.numpy().flatten(), bins=20, alpha=0.5, label="Fake X")
    ax[0].set_title("X Distribution")
    ax[0].legend()

    ax[1].hist(ys_real.numpy().flatten(), bins=20, alpha=0.5, label="Real Y")
    ax[1].hist(ys_fake.numpy().flatten(), bins=20, alpha=0.5, label="Fake Y")
    ax[1].set_title("Y Distribution")
    ax[1].legend()

    if plot_radius:
        rs_fake = sample_generated_y[:, :, 2]
        rs_real = sample_y[:, :, 2]

        ax[-1].hist(rs_real.numpy().flatten(), bins=20, alpha=0.5, label="Real R")
        ax[-1].hist(rs_fake.numpy().flatten(), bins=20, alpha=0.5, label="Fake R")
        ax[-1].set_title("R Distribution")
        ax[-1].legend()

    if plot_distances:
        import torch

        dist_real = torch.cdist(sample_y[:, :, :2], sample_y[:, :, :2]).numpy()
        dist_fake = torch.cdist(
            sample_generated_y[:, :, :2], sample_generated_y[:, :, :2]
        ).numpy()

        ax[-1].hist(dist_real.flatten(), bins=20, alpha=0.5, label="Real Distances")
        ax[-1].hist(dist_fake.flatten(), bins=20, alpha=0.5, label="Fake Distances")
        ax[-1].set_title("Distances")
        ax[-1].legend()

    if return_fig:
        return fig


def plot_sample_figures(
    generator,
    discriminator,
    dataset=None,
    sample_y=None,
    n=5,
    plot_radius=True,
    return_fig=False,
):
    sample_generated_y = generator(dataset[n][0]).detach()

    if sample_y is None:
        try:
            sample_y = dataset[n][1].unsqueeze(0)
        except TypeError:
            raise ValueError("Either 'sample_y' or 'dataset' must be defined!")

    # Illustrate the point cloud

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Set color palette for pyplot
    plt.set_cmap("viridis")

    plot_pointcloud(sample_y.squeeze(0).cpu(), ax=ax[0], plot_radius=plot_radius)

    plot_pointcloud(
        sample_generated_y.squeeze(0).cpu().detach(), ax=ax[1], plot_radius=plot_radius
    )
    # Discriminator predictions on the generated data and the real data

    real_preds = discriminator(sample_y).detach().cpu()
    preds = discriminator(sample_generated_y).detach().cpu()

    ax[0].set_title("Real, discriminator pred: {:.2f}".format(real_preds[0].item()))
    ax[1].set_title("Generated, discriminator pred: {:.2f}".format(preds[0].item()))

    ax[0].axis("equal")
    ax[1].axis("equal")

    [a.set_xlim(-0.1, 1.1) for a in ax]
    [a.set_ylim(-0.1, 1.1) for a in ax]

    if not return_fig:
        plt.show()
    else:
        return fig

    # Plot distribution of the generated data
    # sns.histplot(sample_generated_y[0].detach().numpy()[:,0], bins="auto")
    # plt.title("X")
    # plt.show()

    # sns.histplot(sample_generated_y[0].detach().numpy()[:,1], bins="auto")
    # plt.title("Y")
    # plt.show()

    # sns.histplot(sample_generated_y[0].detach().numpy()[:,2], bins="auto")
    # plt.title("R")
    # plt.show()
