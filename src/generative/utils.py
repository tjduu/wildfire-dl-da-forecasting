"""This module contains functions and utilities used throughout the generative package."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error


def best_obs_mse_image(
    autoencoder: nn.Module,
    obs_dataset_path: str,
    num_generated: int = 500,
    latent_dim: int = 32,
    device: str = "cpu",
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """
    Perform data assimilation using a pre-trained VAE.

    Args:
        autoencoder (nn.Module): The pre-trained VAE model.
        background_dataset_path (str): Path to the file containing background images.
        num_generated (int): Number of images to generate from the latent space.
        latent_dim (int): Dimension of the latent space.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        lowest_mse (float): The lowest MSE value found.
        best_generated_image (np.ndarray): The generated image with the lowest MSE.
        best_background_image (np.ndarray): The background image with the lowest MSE.
        best_background_index (int): The index of the background image with the lowest MSE.
    """
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # Load the background images
    obs_images = np.load(obs_dataset_path)
    obs_images = obs_images.squeeze()  # Ensure images have correct dimensions

    # Generate images from latent space
    z = torch.randn(num_generated, latent_dim).to(device)
    with torch.no_grad():
        generated_images = autoencoder.decoder(z).cpu().numpy()

    # Initialize variables to store the lowest MSE and corresponding images
    lowest_mse = float("inf")
    best_generated_image = None
    best_obs_image = None
    best_obs_index = None

    # Iterate through generated images and compare with background images
    for i in range(num_generated):
        generated_image = generated_images[i].squeeze()

        for j in range(len(obs_images)):
            obs_image = obs_images[j].squeeze()

            # Compute MSE
            mse = mean_squared_error(obs_image, generated_image)

            # Update the lowest MSE and corresponding images
            if mse < lowest_mse:
                lowest_mse = mse
                best_generated_image = generated_image
                best_obs_image = obs_image
                best_obs_index = j

    # Plot the best matching images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(best_generated_image, cmap="viridis")
    ax[0].set_title("Best Generated Image")
    ax[0].axis("off")

    ax[1].imshow(best_obs_image, cmap="viridis")
    ax[1].set_title("Best Observation Image")
    ax[1].axis("off")

    plt.show()

    return lowest_mse, best_generated_image, best_obs_image, best_obs_index


def get_device():
    """Return the device (GPU or CPU) that the user's environment is running on/in."""
    device = "cpu"
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = "cuda"
        print(f"Cuda installed! Running on GPU! (device = {device})")
    else:
        print(f"No GPU available! (device = {device})")

    return device


def sequential_undersample_3d_arr(arr, sequence_jump, _print: bool = True):
    """
    Sequentially undersamples a 3D array.

    Parameters:
        arr (np.ndarray): The 3D array to undersample.
        sequence_jump (int): The interval at which to undersample.
        _print (bool): Flag to print the shape of the undersampled array.

    Returns:
        np.ndarray: The undersampled array.
    """
    assert len(arr.shape) == 3
    sequence_jump = max(1, sequence_jump)  # avoids error at 0.

    undersampled_arr = arr[0::sequence_jump, :, :]

    if _print:
        print("\n** (undersampled) data info **")
        print(f"X_train_undersampled: {undersampled_arr.shape}")

    return undersampled_arr


def sequential_train_val_split(
    train_data,
    sequence_jump: int,
    start_offset: int = 2,
    jump_multiplier: int = 3,
    _print: bool = True,
):
    """Creates a training and validation split for sequential data using undersampling.

    The function assumes that the dimensions are (num_images, height (H), width (W)).

    Parameters:
        train_data (np.ndarray): The dataset to split.
        sequence_jump (int): The base jump for creating validation indices.
        start_offset (int): The start offset for the first validation index.
        jump_multiplier (int): Multiplier to apply to the sequence jump for validation indexing.
        _print (bool): Flag to print the shapes of the training and validation datasets.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing training and validation datasets.
    """
    val_idx = []
    for i in range(
        (sequence_jump * start_offset) + 1,
        train_data.shape[0],
        sequence_jump * jump_multiplier,
    ):
        val_idx.append(i)

    train_idx = [i for i in range(train_data.shape[0]) if i not in val_idx]

    X_train = train_data[train_idx]
    X_val = train_data[val_idx]

    if _print:
        print(f"\n** (split) data info (w/ jump={sequence_jump})**")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")

    return X_train, X_val
