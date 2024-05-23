"""
This module contains functions for plotting reconstructed images from
generative fire prediction models.

Functions:
    plot_batch_recon_images: Plots reconstructed images from a batch
    using a specified model.
"""

from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


def plot_single_raw_and_dataset_idx_image(
    data: Union[List[Any], np.ndarray],
    dataset: Dataset,
    idx: int = 1,
    figsize: Tuple[int, int] = (7, 4),
    cmap: str = "viridis",
    fontsize: int = 10,
):
    """
    Plot a single image from raw data and a dataset at the specified index.

    Parameters
    ----------
    data : Any
        The raw data containing images.
    dataset : Dataset
        The dataset object that supports indexing and returns a tensor image.
    idx : int, optional
        The index of the image to be plotted, by default 1.
    figsize : Tuple[int, int], optional
        The size of the figure (width, height) in inches, by default (7, 4).
    cmap : str, optional
        The color map used to display the image, by default "hot".
    fontsize : int, optional
        Font size for the title of the plots, by default 10.

    Returns
    -------
    None
        This function does not return anything but displays a matplotlib plot.
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    axs[0].imshow(data[idx], cmap=cmap)
    axs[0].axis("off")
    axs[1].imshow(dataset.__getitem__(idx).permute(1, 2, 0), cmap=cmap)
    axs[0].set_title(f"raw data @ idx {idx}", fontsize=fontsize)
    axs[1].set_title(f"dataset.__getitem__({idx})", fontsize=fontsize)
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()
