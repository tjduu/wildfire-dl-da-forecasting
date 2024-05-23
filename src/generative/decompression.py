"""
This module contains classes for decompressing data for data assimilation
using different models, such as Variational Autoencoders (VAE) and Principal
Component Analysis (PCA).

Example usage:
    autoencoder = VAE(input_image_dims=(1, 256, 256),
                      h_dim1=512,
                      h_dim2=256,
                      h_dim3=128,
                      latent_dims=16,
                      device=DEVICE).to(DEVICE)

    vae_test = VAEDecompressor(model_obj=autoencoder,
                                model_path="model_test.pt",
                                device=DEVICE)
    A = vae_test.encode(x=torch.randn(1, 256*256))
    B = vae_test.decode(x=torch.randn(1, 16))

    pca_test = PCADecompressor(model_path="pca_1_model.pkl")
    A = pca_test.encode(x=torch.randn(1, 256*256))
    B = pca_test.decode(x=torch.randn(1, 1))

    A.shape, B.shape
"""


from abc import abstractmethod
import joblib
import torch


class Decompressor:
    """
    Abstract base class for decompressors.

    Methods:
        load_model(model_path): Load the model from the specified path.
        encode(x): Encode the input data.
        decode(x): Decode the encoded data.
    """
    def __init__(self):
        self.model = None

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass


class VAEDecompressor(Decompressor):
    """
    Decompressor using a Variational Autoencoder (VAE).

    Args:
        model_path (str): Path to the VAE model file.
        model_obj (torch.nn.Module): VAE model object.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Methods:
        load_model(model_path, model_obj, device): Load the VAE model from the
        specified path.
        encode(x): Encode the input data using the VAE.
        decode(x): Decode the encoded data using the VAE.
    """
    def __init__(self, model_path, model_obj, device):
        super().__init__()

        self.model_path = model_path
        self.model_obj = model_obj
        self.device = device

        self.model = self.load_model(
            model_path=model_path, model_obj=self.model_obj, device=self.device
        )

    def load_model(self, model_path, model_obj, device):
        """
        Load the VAE model from the specified path.

        Args:
            model_path (str): Path to the VAE model file.
            model_obj (torch.nn.Module): VAE model object.
            device (str): Device to load the model on ('cpu' or 'cuda').

        Returns:
            torch.nn.Module: Loaded VAE model.
        """
        try:
            model_state = torch.load(model_path,
                                     map_location=torch.device(device))
            model_obj.load_state_dict(model_state["model_state_dict"])
            model_obj.to(device)
            model_obj.eval()

            return model_obj

        except KeyError:
            raise ValueError("Model state dict key not found in the model")

    def encode(self, x):
        """
        Encode the input data using the VAE.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Encoded data tensor.
        """
        return self.model.encode(x.to(self.device))

    def decode(self, x):
        """
        Decode the encoded data using the VAE.

        Args:
            x (torch.Tensor): Encoded data tensor.

        Returns:
            torch.Tensor: Decoded data tensor.
        """
        return self.model.decode(x.to(self.device))


class PCADecompressor(Decompressor):
    """
    Decompressor using Principal Component Analysis (PCA).

    Args:
        model_path (str): Path to the PCA model file.

    Methods:
        load_model(model_path): Load the PCA model from the specified path.
        encode(x): Encode the input data using the PCA model.
        decode(x): Decode the encoded data using the PCA model.
    """
    def __init__(self, model_path):
        super().__init__()

        self.model_path = model_path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the PCA model from the specified path.

        Args:
            model_path (str): Path to the PCA model file.

        Returns:
            sklearn.decomposition.PCA: Loaded PCA model.
        """
        model_obj = joblib.load(model_path)
        return model_obj

    def encode(self, x):
        """
        Encode the input data using the PCA model.

        Args:
            x (numpy.ndarray): Input data array.

        Returns:
            numpy.ndarray: Encoded data array.
        """
        return self.model.transform(x)

    def decode(self, x):
        """
        Decode the encoded data using the PCA model.

        Args:
            x (numpy.ndarray): Encoded data array.

        Returns:
            numpy.ndarray: Decoded data array.
        """
        return self.model.inverse_transform(x)
