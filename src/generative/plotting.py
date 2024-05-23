"""
This module contains functions for plotting reconstructed images from
generative fire prediction models.

Functions:
    plot_batch_recon_images: Plots reconstructed images from a batch
    using a specified model.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def plot_batch_recon_images(
    model: nn.Module,
    data_loader: DataLoader,
    num_images: int = 5,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (17, 7),
    fontsize: int = 8,
    plot_random: bool = True,
    plot_idxs: Optional[List[int]] = None,
    device: str = "cpu",
) -> None:
    """
    Plot reconstructed images from a batch using a given model.

    Args:
        model (torch.nn.Module): The model to be used for image reconstruction.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the
                                                    batch of images.
        num_images (int, optional): Number of images to plot. Defaults to 5.
        cmap (str, optional): Colormap for displaying images.
                            Defaults to 'viridis'.
        figsize (tuple, optional): Size of the figure for plotting.
                                    Defaults to (17, 7).
        fontsize (int, optional): Font size for the titles. Defaults to 8.
        plot_random (bool, optional): Whether to plot random images from the
                                      batch. Defaults to True.
        plot_idxs (list, optional): List of indices to plot if plot_random is
                                    False. Defaults to None.
        device (str, optional): Device to perform computations on.
                                Defaults to 'cpu'.
    """
    num_images = min(data_loader.batch_size, num_images)
    img_batch = next(iter(data_loader))  # first batch of images.
    model.eval()

    fig, ax = plt.subplots(nrows=3, ncols=num_images, figsize=figsize)

    if plot_random:
        idxs = torch.randint(0, data_loader.batch_size, (num_images,))
    else:
        idxs = plot_idxs if plot_idxs is not None else [0, 1, 2]

    for n, idx in enumerate(idxs):
        img = img_batch[idx].float().unsqueeze(0).to(device)
        recon, _ = model(img)

        img = img.detach().cpu().squeeze()
        recon = recon.detach().cpu().squeeze()

        ax[0, n].imshow(img.numpy(), cmap=cmap)
        ax[0, n].set_title(f"raw {idx} idx image", fontsize=fontsize)
        ax[0, n].axis("off")

        ax[1, n].imshow(recon.numpy(), cmap=cmap)
        ax[1, n].set_title(f"recon {idx} idx image", fontsize=fontsize)
        ax[1, n].axis("off")

        ax[2, n].imshow((img - recon).numpy(), cmap=cmap)
        ax[2, n].set_title("diff (raw - recon)", fontsize=fontsize)
        ax[2, n].axis("off")

    plt.tight_layout()
    plt.show()
