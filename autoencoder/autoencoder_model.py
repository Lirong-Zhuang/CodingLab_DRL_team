import torch
import torch.nn as nn


class GridEncoderEnv5(nn.Module):
    def __init__(self, in_channels=5, feature_dim=64, feature_channels=None):
        super().__init__()
        if feature_channels is not None:
            feature_dim = feature_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.feature = nn.Sequential(
            nn.Linear(32 * 3 * 3, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.feature(x)


class GridDecoderEnv5(nn.Module):
    def __init__(self, out_channels=5, feature_dim=64, feature_channels=None):
        super().__init__()
        if feature_channels is not None:
            feature_dim = feature_channels
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32 * 3 * 3),
            nn.ReLU(),
            nn.Unflatten(1, (32, 3, 3)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class GridAutoencoderEnv5(nn.Module):
    def __init__(self, in_channels=5, feature_dim=64, feature_channels=None):
        super().__init__()
        if feature_channels is not None:
            feature_dim = feature_channels
        self.encoder = GridEncoderEnv5(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        self.decoder = GridDecoderEnv5(
            out_channels=in_channels,
            feature_dim=feature_dim,
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x


# Backward-compatible aliases for older training scripts.
Encoder = GridEncoderEnv5
Decoder = GridDecoderEnv5
Autoencoder = GridAutoencoderEnv5
