from matplotlib import pyplot as plt
from scipy.stats import kstest


def plot_pointcloud(pointcloud, ax=None, plot_radius=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    xs = pointcloud[:, 0]
    ys = pointcloud[:, 1]

    if plot_radius:
        rs = pointcloud[:, 2]

    if plot_radius:
        ax.scatter(xs, ys, c=rs, s=rs * 30, alpha=0.5)
    else:
        ax.scatter(
            xs, ys, alpha=0.5, s=10
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

    xs_real = sample_y[:, :, 0].numpy().flatten()
    ys_real = sample_y[:, :, 1].numpy().flatten()

    xs_fake = sample_generated_y[:, :, 0].numpy().flatten()
    ys_fake = sample_generated_y[:, :, 1].numpy().flatten()

    # Kolmogorov smirnoff test

    ks_test_x = kstest(xs_real, xs_fake, alternative="two-sided")
    ks_test_y = kstest(ys_real, ys_fake, alternative="two-sided")

    # Plot the distributions of the different variables in different figures
    ax[0].hist(xs_real, bins=5, alpha=0.5, label="Real X")
    ax[0].hist(xs_fake, bins=5, alpha=0.5, label="Fake X")
    ax[0].set_title("X Distribution, ks: {:.2f}, p: {:.2f}".format(ks_test_x.statistic, ks_test_x.pvalue))
    ax[0].legend()

    ax[1].hist(ys_real, bins=5, alpha=0.5, label="Real Y")
    ax[1].hist(ys_fake, bins=5, alpha=0.5, label="Fake Y")
    ax[1].set_title("Y Distribution, ks: {:.2f}, p: {:.2f}".format(ks_test_y.statistic, ks_test_y.pvalue))
    ax[1].legend()

    if plot_radius:
        rs_real = sample_y[:, :, 2]
        rs_fake = sample_generated_y[:, :, 2]

        ax[2].hist(rs_real, bins=20, alpha=0.5, label="Real R")
        ax[2].hist(rs_fake, bins=20, alpha=0.5, label="Fake R")
        ax[2].set_title("R Distribution")
        ax[2].legend()

    if plot_distances:
        import torch

        dist_real = torch.cdist(sample_y[:, :, :2], sample_y[:, :, :2], p=2)
        dist_fake = torch.cdist(
            sample_generated_y[:, :, :2], sample_generated_y[:, :, :2], p=2
        )

        k = 3  # Number of nearest neighbors to consider
        dist_real = torch.topk(dist_real, k=k, dim=2, largest=False).values
        dist_fake = torch.topk(dist_fake, k=k, dim=2, largest=False).values

        dist_real = dist_real.numpy().flatten()
        dist_fake = dist_fake.numpy().flatten()

        ks_test_f = kstest(dist_real, dist_fake, alternative="two-sided")

        ax[-1].hist(dist_real, bins="auto", alpha=0.5, label="Real Distances")
        # Get the bins and set for the fake distances

        ax[-1].hist(dist_fake, bins="auto", alpha=0.5, label="Fake Distances")
        ax[-1].set_title("Distances, ks: {:.2f}, p: {:.2f}".format(ks_test_f.statistic, ks_test_f.pvalue))
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
