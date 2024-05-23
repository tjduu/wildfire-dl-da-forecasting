""""""

from abc import abstractmethod
import joblib
import torch

"""
example usage:

autoencoder = VAE(input_image_dims=(1, 256, 256),
                  h_dim1=512,
                  h_dim2=256,
                  h_dim3=128,
                  latent_dims=16,
                  device=DEVICE).to(DEVICE)

vae_test = VAEDecompressor(model_obj=autoencoder, model_path="model_test.pt", device=DEVICE)
A = vae_test.encode(x=torch.randn(1, 256*256))
B = vae_test.decode(x=torch.randn(1, 16))

pca_test = PCADecompressor(model_path="pca_1_model.pkl")
A = pca_test.encode(x=torch.randn(1, 256*256))
B = pca_test.decode(x=torch.randn(1, 1))

A.shape, B.shape
"""


class Decompressor:
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
    def __init__(self, model_path, model_obj, device):
        super().__init__()

        self.model_path = model_path
        self.model_obj = model_obj
        self.device = device

        self.model = self.load_model(
            model_path=model_path, model_obj=self.model_obj, device=self.device
        )

    def load_model(self, model_path, model_obj, device):
        try:
            model_state = torch.load(model_path, map_location=torch.device(device))
            model_obj.load_state_dict(model_state["model_state_dict"])
            model_obj.to(device)
            model_obj.eval()

            return model_obj

        except KeyError:
            raise ValueError("Model state dict key not found in the loaded model file")

    def encode(self, x):
        return self.model.encode(x.to(self.device))

    def decode(self, x):
        return self.model.decode(x.to(self.device))


class PCADecompressor(Decompressor):
    # non CPU/GPU --> dont need device.
    def __init__(self, model_path):
        super().__init__()

        self.model_path = model_path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model_obj = joblib.load(model_path)
        return model_obj

    def encode(self, x):
        return self.model.transform(x)

    def decode(self, x):
        return self.model.inverse_transform(x)
