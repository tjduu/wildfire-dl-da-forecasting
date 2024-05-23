""""""

import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Define the data_assimilation function
def best_obs_mse_image(autoencoder, obs_dataset_path, num_generated=500, latent_dim=32, device='cpu'):
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
    lowest_mse = float('inf')
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
    ax[0].imshow(best_generated_image, cmap='viridis')
    ax[0].set_title("Best Generated Image")
    ax[0].axis('off')
 
    ax[1].imshow(best_obs_image, cmap='viridis')
    ax[1].set_title("Best Observation Image")
    ax[1].axis('off')
 
    plt.show()
 
    return lowest_mse, best_generated_image, best_obs_image, best_obs_index


# # Test the data_assimilation function
# lowest_mse, best_generated_image, best_obs_image, best_obs_index = best_obs_mse_image(
#     vae, obs_dataset, num_generated=500, latent_dim=latent_dim, device=device
# )
 
# print(f'Lowest MSE: {lowest_mse}')
# print(f'Best Observation Image Index: {best_obs_index}')


def get_device():
    """"""
    device = "cpu"
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = "cuda"
    else:
        print("No GPU available!")
    return device


def sequential_undersample_3d_arr(arr, sequence_jump, _print:bool=True):
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

def sequential_train_val_split(train_data, sequence_jump:int, start_offset:int=2, jump_multiplier:int=3, _print:bool=True):
    """assumes item number / time is first dim then HxW.
    1 test: assert train_data.shape[0] == len(train_idx) + len(val_idx)
    """
    val_idx = []
    for i in range((sequence_jump * start_offset) + 1, train_data.shape[0], sequence_jump * jump_multiplier):
        val_idx.append(i)

    train_idx = [i for i in range(train_data.shape[0]) if i not in val_idx]

    X_train = train_data[train_idx]
    X_val = train_data[val_idx]

    if _print:
        print(f"\n** (split) data info (w/ jump={sequence_jump})**")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")   

    return X_train, X_val 