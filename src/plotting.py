from matplotlib import pyplot as plt


def plot_pointcloud(pointcloud, ax=None, plot_radius=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    xs = pointcloud[:, 0]
    ys = pointcloud[:, 1]
    rs = pointcloud[:, 2]

    if plot_radius:
        ax.scatter(xs, ys, c=rs, s=rs * 30, alpha=0.5)
    else:
        ax.scatter(
            xs, ys, c=rs, alpha=0.5
        )  # TODO: Size of the points is not correct in comparison to the grid size

    return ax


def plot_sample_figures(
    generator,
    discriminator,
    dataset=None,
    sample_y=None,
    n=5,
    plot_radius=True,
    return_fig=False,
):
    sample_generated_y = generator(dataset[:][0][0:n])

    if sample_y is None:
        try:
            sample_y = dataset[:][1][0:n]
        except TypeError:
            raise ValueError("Either 'sample_y' or 'dataset' must be defined!")

    # Illustrate the point cloud

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    plot_pointcloud(sample_y[0], ax=ax[0], plot_radius=plot_radius)

    plot_pointcloud(
        sample_generated_y[0].detach().numpy(), ax=ax[1], plot_radius=plot_radius
    )

    # Discriminator predictions on the generated data and the real data

    real_preds = discriminator(sample_y)
    preds = discriminator(sample_generated_y)

    ax[0].set_title("Real, discriminator pred: {:.2f}".format(real_preds[0].item()))
    ax[1].set_title("Generated, discriminator pred: {:.2f}".format(preds[0].item()))

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
