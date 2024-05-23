""""""
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter


def get_latent_dim(model):
    """
    Get the latent dimension of the VAE model.

    Args:
        model (nn.Module): The VAE model.
    
    Returns:
        latent_dim (int): The latent dimension of the model.
    """
    return model._mu.out_features

def generate_and_filter_images(autoencoder, obs_dataset, device='cpu', num_samples=500, pixel_ratio_range=(5, 14), threshold=0.2, num_images=10, print_info=False):
    """
    Generate images from random latent vectors and filter them based on pixel ratio criteria.

    Args:
        autoencoder (nn.Module): The VAE model.
        obs_dataset (np.ndarray): The dataset of observation images.
        device (str): The device to run the model on ('cpu' or 'cuda').
        num_samples (int): Number of images to generate from the latent space.
        pixel_ratio_range (tuple): Range of pixel activation ratio to filter generated images.
        threshold (float): The threshold value for binarizing the generated images.
        num_images (int): Number of "top" images to generate and display.
        print_info (bool): Whether to print debugging information.

    Returns:
        filtered_generated_images (list): List of generated images that meet the pixel ratio criteria.
        best_generated_image (np.ndarray): The generated image with the lowest MSE.
        best_obs_image (np.ndarray): The observation image with the lowest MSE.
        best_obs_index (int): The index of the observation image with the lowest MSE.
        lowest_mse (float): The lowest MSE value found.
        top_images (list): List of the generated images with the lowest MSE.
        top_mses (list): List of the MSE values of the top generated images.

    Example:
        filtered_generated_images, best_generated_image, best_obs_image, best_obs_index, lowest_mse, top_images, top_mses = generate_and_filter_images(
            model,
            obs_dataset,
            device='cpu',
            num_samples=8000,
            pixel_ratio_range=(5, 14),
            num_images=50,
            threshold=0.70 # Adjust threshold based on the dataset
        )
    """
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    latent_dim = get_latent_dim(autoencoder)

    # Generate images from latent space
    z = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        generated_images = autoencoder.decoder(z).cpu().numpy()
    
    # Reshape generated images to 2D
    generated_images = generated_images.reshape((num_samples, 256, 256))

    # Apply Gaussian filter to smooth images
    smoothed_images = [gaussian_filter(img, sigma=1) for img in generated_images]

    # Filtered images based on pixel ratio criteria
    filtered_generated_images = []

    for i in range(num_samples):
        generated_image = smoothed_images[i].squeeze()

        # Binarize the generated image with the specified threshold
        binary_image = np.where(generated_image >= threshold, 1, 0)

        # Calculate pixel ratio
        total_pixels = np.prod(binary_image.shape)
        active_pixels = np.sum(binary_image == 1)
        pixel_ratio = (active_pixels / total_pixels) * 100

        if print_info:
        # Debugging prints
            print(f"Generated Image {i + 1}/{num_samples}")
            print(f"Threshold: {threshold}")
            print(f"Total Pixels: {total_pixels}")
            print(f"Active Pixels: {active_pixels}")
            print(f"Pixel Ratio: {pixel_ratio:.2f}%")

        # Check if the pixel ratio is within the specified range
        if pixel_ratio_range[0] <= pixel_ratio <= pixel_ratio_range[1]:
            filtered_generated_images.append(binary_image)

    # Check if filtered images are 0 or not
    if len(filtered_generated_images) == 0:
        print('No images meet the pixel ratio criteria.')
        return None

    total_valid_generated_images = len(filtered_generated_images)
    print(f"Total Valid Generated Images: {total_valid_generated_images}")

    # Initialize variables to store the lowest MSE and corresponding images
    lowest_mse = float('inf')
    highest_ssim = -1
    best_generated_image = None
    best_obs_image = None
    best_obs_index = None

    # Iterate through generated images and compare with observation images
    for i in range(total_valid_generated_images):
        generated_image = filtered_generated_images[i]

        for j in range(len(obs_dataset)):
            obs_image = obs_dataset[j].squeeze()

            # Compute MSE
            mse = mean_squared_error(obs_image, generated_image)
            # Compute SSIM
            ssim_value = ssim(obs_image, generated_image, data_range=generated_image.max() - generated_image.min())

            # Update the best image based on combined criteria
            if mse < lowest_mse and ssim_value > highest_ssim:
                lowest_mse = mse
                highest_ssim = ssim_value
                best_generated_image = generated_image
                best_obs_image = obs_image
                best_obs_index = j
    
    # Plot the best image with the obs image side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(best_generated_image, cmap='viridis')
    ax[0].set_title("Best Generated Image")
    ax[0].axis('off')

    ax[1].imshow(best_obs_image, cmap='viridis')
    ax[1].set_title("Best Observation Image")
    ax[1].axis('off')
    plt.show()

    # Compare filtered images with the best generated image
    mse_list = []
    for generated_image in filtered_generated_images:
        mse = mean_squared_error(best_generated_image, generated_image)
        mse_list.append(mse)
    
    # Get the top 100 images
    top_indices = np.argsort(mse_list)[:num_images]
    top_images = [filtered_generated_images[i] for i in top_indices]
    top_mses = [mse_list[i] for i in top_indices]

    # Plot top images if less than or equal to 100, else plot the first 100
    images_to_plot = min(num_images, 100)
    num_rows = (images_to_plot + 9) // 10  # Ensure 10 images per row
    fig, ax = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
    
    for i in range(images_to_plot):
        row = i // 10
        col = i % 10
        ax[row, col].imshow(top_images[i], cmap='viridis')
        ax[row, col].set_title(f"MSE: {top_mses[i]:.2f}")
        ax[row, col].axis('off')
    
    # Hide any unused subplots
    for j in range(images_to_plot, num_rows * 10):
        row = j // 10
        col = j % 10
        fig.delaxes(ax[row, col])
    
    plt.tight_layout()
    plt.show()

    return filtered_generated_images, best_generated_image, best_obs_image, best_obs_index, lowest_mse, top_images, top_mses


def get_device():
    """"""
    device = "cpu"
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = "cuda"
        print(f"Cuda installed! Running on GPU! (device = {device})")
    else:
        print(f"No GPU available! (device = {device})")

    return device


def sequential_undersample_3d_arr(arr, sequence_jump, _print: bool = True):
    """
    test cases:
    arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])

    arr[0:15, :, :], arr[0:0, :, :], arr[0:2, :, :]

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
    """assumes item number / time is first dim then HxW.
    1 test: assert train_data.shape[0] == len(train_idx) + len(val_idx)
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
