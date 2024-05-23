""""""
import torch
from torch.utils.data import Dataset


class WildfireImageDataSet(Dataset):
    def __init__(self, data, transform):
        """Initialisation."""
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        """Get raw data using idx and apply transforms to output a processed image tensor."""
        image = self.data[idx] 
        if self.transform:
            image = self.transform(image)
        assert isinstance(image, torch.Tensor)

        return image

    def __len__(self):
        return len(self.data)