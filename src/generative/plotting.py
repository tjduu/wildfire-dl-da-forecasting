""""""

import matplotlib.pyplot as plt

def plot_batch_recon_images(
    model,
    data_loader,
    num_images: int = 5,
    cmap: str = "viridis",
    figsize: tuple = (17, 7),
    fontsize: int = 8,
    plot_random: bool = True,
    plot_idxs: list = [0, 1, 2],
    device: str = "cpu",
):
    """"""
    num_images = min(data_loader.batch_size, num_images)

    img_batch = next(iter(data_loader))  # first batch of images.

    model.eval()

    fig, ax = plt.subplots(nrows=3, ncols=num_images, figsize=figsize)

    if plot_random:
        idxs = torch.randint(0, data_loader.batch_size, (num_images,))
        print(f"random idxs = {idxs}")
    else:
        idxs = plot_idxs

    for n, idx in enumerate(idxs):
        img = img_batch[idx].float().unsqueeze(0).to(device)
        recon, _ = model(img)

        img = img.detach().cpu().squeeze()
        recon = recon.detach().cpu().squeeze()

        ax[0, n].imshow(img, cmap=cmap)
        ax[0, n].set_title(f"raw {idx} idx image.", fontsize=fontsize)
        ax[0, n].axis("off")

        ax[1, n].imshow(recon, cmap=cmap)
        ax[1, n].set_title(f"recon {idx} idx image.", fontsize=fontsize)
        ax[1, n].axis("off")

        ax[2, n].imshow(img - recon, cmap=cmap)
        ax[2, n].set_title(f"diff (raw - recon).", fontsize=fontsize)
        ax[2, n].axis("off")

    plt.tight_layout()
    plt.show()
