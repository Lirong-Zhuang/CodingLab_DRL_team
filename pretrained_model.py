import torch
import torch.nn as nn
from torchvision.models import resnet18


class RainbowDuelingDistributionalHead(nn.Module):
    def __init__(self, feature_dim, act_dim, num_atoms):
        super().__init__()
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_atoms),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, act_dim * num_atoms),
        )

    def forward(self, features):
        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(
            -1,
            self.act_dim,
            self.num_atoms,
        )
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(logits, dim=-1)


class LeNet5Encoder(nn.Module):
    """LeNet-5-style encoder adapted for small 5x5 grid observations."""

    def __init__(self, in_channels, feature_dim=84):
        super().__init__()
        self.feature_dim = feature_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=2),
            nn.Tanh(),
            nn.Flatten(),
        )

        self.feature = nn.Sequential(
            nn.Linear(16, 120),
            nn.Tanh(),
            nn.Linear(120, feature_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.feature(x)


class AlexNet8Encoder(nn.Module):
    """AlexNet-style 8-layer encoder adapted for small 5x5 grid observations."""

    def __init__(self, in_channels, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

        self.feature = nn.Sequential(
            nn.Linear(64 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.feature(x)


class ResNet18Encoder(nn.Module):
    """ResNet18 encoder adapted for small grid observations."""

    def __init__(self, in_channels, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.scene_encoder = backbone
        self.feature = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.scene_encoder(x)
        return self.feature(x)


class RainbowLeNet5QNetwork(nn.Module):
    def __init__(self, in_channels, act_dim, num_atoms, feature_dim=84):
        super().__init__()
        self.encoder = LeNet5Encoder(in_channels, feature_dim)
        self.q_head = RainbowDuelingDistributionalHead(
            feature_dim,
            act_dim,
            num_atoms,
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.q_head(features)

    def reset_noise(self):
        pass


class RainbowResNet18QNetwork(nn.Module):
    def __init__(self, in_channels, act_dim, num_atoms, feature_dim=128):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels, feature_dim)
        self.q_head = RainbowDuelingDistributionalHead(
            feature_dim,
            act_dim,
            num_atoms,
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.q_head(features)

    def reset_noise(self):
        pass


class RainbowAlexNet8QNetwork(nn.Module):
    def __init__(self, in_channels, act_dim, num_atoms, feature_dim=256):
        super().__init__()
        self.encoder = AlexNet8Encoder(in_channels, feature_dim)
        self.q_head = RainbowDuelingDistributionalHead(
            feature_dim,
            act_dim,
            num_atoms,
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.q_head(features)

    def reset_noise(self):
        pass
