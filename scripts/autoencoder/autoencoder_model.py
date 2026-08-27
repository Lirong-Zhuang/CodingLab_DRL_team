import torch
import torch.nn as nn

from algorithms.pretrained_model import AlexNet8Encoder, LeNet5Encoder, ResNet18Encoder


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


class GridEncoderEnv9(GridEncoderEnv5):
    def __init__(self, in_channels=6, feature_dim=64, feature_channels=None):
        super().__init__(
            in_channels=in_channels,
            feature_dim=feature_dim,
            feature_channels=feature_channels,
        )


class GridDecoderEnv9(GridDecoderEnv5):
    def __init__(self, out_channels=6, feature_dim=64, feature_channels=None):
        super().__init__(
            out_channels=out_channels,
            feature_dim=feature_dim,
            feature_channels=feature_channels,
        )


class GridAutoencoderEnv9(nn.Module):
    def __init__(self, in_channels=6, feature_dim=64, feature_channels=None):
        super().__init__()
        if feature_channels is not None:
            feature_dim = feature_channels
        self.encoder = GridEncoderEnv9(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        self.decoder = GridDecoderEnv9(
            out_channels=in_channels,
            feature_dim=feature_dim,
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x


class GridEncoderEnv11(GridEncoderEnv5):
    def __init__(self, in_channels=7, feature_dim=64, feature_channels=None):
        super().__init__(
            in_channels=in_channels,
            feature_dim=feature_dim,
            feature_channels=feature_channels,
        )


class GridDecoderEnv11(GridDecoderEnv5):
    def __init__(self, out_channels=7, feature_dim=64, feature_channels=None):
        super().__init__(
            out_channels=out_channels,
            feature_dim=feature_dim,
            feature_channels=feature_channels,
        )


class GridAutoencoderEnv11(nn.Module):
    def __init__(self, in_channels=7, feature_dim=64, feature_channels=None):
        super().__init__()
        if feature_channels is not None:
            feature_dim = feature_channels
        self.encoder = GridEncoderEnv11(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        self.decoder = GridDecoderEnv11(
            out_channels=in_channels,
            feature_dim=feature_dim,
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x


class FlexibleGridAutoencoder(nn.Module):
    def __init__(self, encoder, out_channels, feature_dim=64, feature_channels=None):
        super().__init__()
        if feature_channels is not None:
            feature_dim = feature_channels
        self.encoder = encoder
        self.decoder = GridDecoderEnv5(
            out_channels=out_channels,
            feature_dim=feature_dim,
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def build_autoencoder(in_channels, feature_dim=64, encoder_type="cnn", feature_channels=None):
    if feature_channels is not None:
        feature_dim = feature_channels

    encoder_type = encoder_type.lower()
    encoder_classes = {
        "lenet5": LeNet5Encoder,
        "alexnet8": AlexNet8Encoder,
        "resnet18": ResNet18Encoder,
    }

    if encoder_type == "cnn":
        if in_channels == 5:
            return GridAutoencoderEnv5(in_channels=in_channels, feature_dim=feature_dim)
        if in_channels == 6:
            return GridAutoencoderEnv9(in_channels=in_channels, feature_dim=feature_dim)
        if in_channels == 7:
            return GridAutoencoderEnv11(in_channels=in_channels, feature_dim=feature_dim)
        encoder = GridEncoderEnv5(in_channels=in_channels, feature_dim=feature_dim)
    elif encoder_type in encoder_classes:
        encoder = encoder_classes[encoder_type](in_channels, feature_dim=feature_dim)
    else:
        raise ValueError(
            "encoder_type must be one of ['cnn', 'lenet5', 'alexnet8', 'resnet18'], "
            f"got {encoder_type!r}."
        )

    return FlexibleGridAutoencoder(
        encoder=encoder,
        out_channels=in_channels,
        feature_dim=feature_dim,
    )


# Backward-compatible aliases for older training scripts.
Encoder = GridEncoderEnv5
Decoder = GridDecoderEnv5
Autoencoder = GridAutoencoderEnv5
