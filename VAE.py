""""""

import torch
import torch.nn as nn


class Reshape(nn.Module):
    """"""

    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):
    """"""

    def __init__(self, input_image_dims, h_dim1, h_dim2, h_dim3, latent_dims, device):
        super().__init__()

        # inputs.
        self.input_image_dims = input_image_dims
        self.c, self.h, self.w = input_image_dims
        self.h_dim1 = h_dim1
        self.h_dim2 = h_dim2
        self.h_dim3 = h_dim3
        self.latent_dims = latent_dims
        self.device = device

        self.activation = nn.ReLU()
        self.distribution = torch.distributions.Normal(0, 1)

        self.encoder = nn.Sequential(
            nn.Linear(self.c * self.h * self.w, h_dim1),
            self.activation,
            nn.Linear(h_dim1, h_dim2),
            self.activation,
            nn.Linear(h_dim2, h_dim3),
            self.activation,
        )

        self._mu = nn.Linear(h_dim3, self.latent_dims)
        self._logvar = nn.Linear(h_dim3, self.latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dims, h_dim3),
            self.activation,
            nn.Linear(h_dim3, h_dim2),
            self.activation,
            nn.Linear(h_dim2, h_dim1),
            self.activation,
            nn.Linear(h_dim1, self.c * self.h * self.w),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """"""
        return self.encoder(x)

    def decode(self, x):
        """"""
        return self.decoder(x)

    def sample_latent_space(self, mu, logvar):
        """"""
        sigma = torch.exp(0.5 * logvar)
        z = mu + sigma * self.distribution.sample(mu.shape).to(self.device)
        kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z, kl_div

    def forward(self, x):
        """"""
        encoded = self.encode(x.flatten(start_dim=1))  # make sure its 1D.

        # get mu and logvar from latent space.
        mu = self._mu(encoded)
        logvar = self._logvar(encoded)

        # reparamaterise trick.
        z, kl_div = self.sample_latent_space(mu, logvar)

        decoded = self.decode(z)

        return decoded, kl_div


class CVAE(nn.Module):
    """"""

    def __init__(
        self,
        input_image_dims,
        latent_dims,
        device,
        kernel_sizes,
        filter_sizes,
        h_dim,
        pool_size: int = 2,
    ):
        super().__init__()

        self.input_image_dims = input_image_dims
        self.c, self.h, self.w = input_image_dims
        self.latent_dims = latent_dims
        self.device = device

        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.h_dim = h_dim
        self.pool_size = pool_size

        self.activation = nn.Mish()
        self.distribution = torch.distributions.Normal(0, 1)

        self.encode = nn.Sequential(
            nn.Conv2d(
                in_channels=self.c,
                out_channels=self.filter_sizes[0],
                kernel_size=self.kernel_sizes[0],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[0]),
            self.activation,
            nn.MaxPool2d(kernel_size=self.pool_size),
            nn.Conv2d(
                in_channels=self.filter_sizes[0],
                out_channels=self.filter_sizes[1],
                kernel_size=self.kernel_sizes[1],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[1]),
            self.activation,
            nn.MaxPool2d(kernel_size=self.pool_size),
            nn.Conv2d(
                in_channels=self.filter_sizes[1],
                out_channels=self.filter_sizes[2],
                kernel_size=self.kernel_sizes[2],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[2]),
            self.activation,
            nn.Flatten(),
            nn.Linear(
                8 * 64 * 64, self.h_dim1
            ),  # TODO: workout flattened() size automatically.
            self.activation,
        )

        self._mu = nn.Linear(self.h_dim, self.latent_dims)
        self._logvar = nn.Linear(self.h_dim, self.latent_dims)

        self.decode = nn.Sequential(
            nn.Linear(self.latent_dims, self.h_dim),
            self.activation,
            nn.Linear(
                self.h_dim, 8 * 64 * 64
            ),  # TODO: workout flattened() size automatically.
            self.activation,
            Reshape(-1, 8, 64, 64),
            nn.Conv2d(
                in_channels=self.filter_sizes[2],
                out_channels=self.filter_sizes[1],
                kernel_size=self.kernel_sizes[2],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[1]),
            nn.Upsample(scale_factor=self.pool_size, mode="bilinear"),
            self.activation,
            nn.Conv2d(
                in_channels=self.filter_sizes[1],
                out_channels=self.filter_sizes[0],
                kernel_size=self.kernel_sizes[1],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(self.filter_sizes[0]),
            nn.Upsample(scale_factor=self.pool_size, mode="bilinear"),
            self.activation,
            nn.Conv2d(
                in_channels=self.filter_sizes[0],
                out_channels=self.c,
                kernel_size=self.kernel_sizes[0],
                stride=1,
                padding="same",
            ),
            # nn.Upsample((256, 256), mode='bilinear'),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """"""
        return self.encoder(x)

    def decode(self, x):
        """"""
        return self.decoder(x)

    def sample_latent_space(self, mu, logvar):
        """"""
        sigma = torch.exp(0.5 * logvar)
        z = mu + sigma * self.distribution.sample(mu.shape).to(self.device)
        kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z, kl_div

    def forward(self, x):
        """"""
        encoded = self.encode(x.flatten(start_dim=1))  # make sure its 1D.

        # get mu and logvar from latent space.
        mu = self._mu(encoded)
        logvar = self._logvar(encoded)

        # reparamaterise trick.
        z, kl_div = self.sample_latent_space(mu, logvar)

        decoded = self.decode(z)

        return decoded, kl_div
