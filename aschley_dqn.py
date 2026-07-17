import numpy as np
import torch
import torch.nn as nn


class AschleyCNNv2:
    """Inference-compatible adapter for dev_ashley_new.DQN_CNN_v2 checkpoints."""

    def __init__(self, env):
        self.env = env
        self.variant = env.variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        initial_obs = env.reset("training")
        self.in_channels = initial_obs.shape[0]
        self.act_dim = 5
        self.epsilon = 0.0
        self.q_network = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, self.act_dim),
        ).to(self.device)

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.act_dim))
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_network(obs_tensor).argmax(dim=1).item())
