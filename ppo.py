import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


# simple PPO agent
class PPO_v1:

    def __init__(self, env):
        self.env = env
        self.variant = env.variant
        self.data_dir = env.data_dir

        

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.network(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns