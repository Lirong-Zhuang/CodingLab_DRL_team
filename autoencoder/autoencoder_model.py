import torch
import torch.nn as nn


class GridEncoderEnv5(nn.Module):
    def __init__(self, in_channels=5, feature_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GridDecoderEnv5(nn.Module):
    def __init__(self, out_channels=5, feature_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class GridAutoencoderEnv5(nn.Module):
    def __init__(self, in_channels=5, feature_channels=64):
        super().__init__()
        self.encoder = GridEncoderEnv5(
            in_channels=in_channels,
            feature_channels=feature_channels,
        )
        self.decoder = GridDecoderEnv5(
            out_channels=in_channels,
            feature_channels=feature_channels,
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x


# Backward-compatible aliases for older training scripts.
Encoder = GridEncoderEnv5
Decoder = GridDecoderEnv5
Autoencoder = GridAutoencoderEnv5
