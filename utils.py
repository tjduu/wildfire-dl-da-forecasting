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
