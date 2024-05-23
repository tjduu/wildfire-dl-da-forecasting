"""
This module contains a custom Dataset class for handling wildfire images with
transformations.

Classes:
    WildfireImageDataSet: A custom Dataset for loading and transforming
    wildfire images.
"""

import torch
from torch.utils.data import Dataset


class WildfireImageDataSet(Dataset):
    """
    A custom Dataset for loading and transforming wildfire images.

    Args:
        data (list or numpy.ndarray): The dataset containing the images.
        transform (callable): A function/transform to apply to the images.

    Methods:
        __getitem__(idx): Get raw data using idx and apply transforms to
        output a processed image tensor.
        __len__(): Return the total number of images in the dataset.
    """
    def __init__(self, data, transform):
        """Initialise the dataset with data and transform.

        Args:
            data (list or numpy.ndarray): The dataset containing the images.
            transform (callable): A function/transform to apply to the images.
        """
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        """Get raw data using idx and apply transforms to output a processed
        image tensor.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        assert isinstance(image, torch.Tensor)

        return image

    def __len__(self):
        """Return the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.data)
