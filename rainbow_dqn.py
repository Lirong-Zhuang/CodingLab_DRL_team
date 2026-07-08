import random
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from pretrained_model import (
    RainbowAlexNet8QNetwork,
    RainbowLeNet5QNetwork,
    RainbowResNet18QNetwork as PretrainedRainbowResNet18QNetwork,
)

# Dueling DQN network architecture
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# Rainbow Q network
class RainbowQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, num_atoms):
        super().__init__()
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            # NoisyLinear(64, 64),
            nn.Linear(64, 64),
            nn.ReLU(),
            # NoisyLinear(64, num_atoms)
            nn.Linear(64, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            # NoisyLinear(64, 64),
            nn.Linear(64, 64),
            nn.ReLU(),
            # NoisyLinear(64, act_dim * num_atoms)
            nn.Linear(64, act_dim * num_atoms)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.act_dim, self.num_atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(logits, dim=-1)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# Qnetwork for cnn and Resbet18
class RainbowResNet18QNetwork(nn.Module):
    def __init__(self, in_channels, act_dim, num_atoms):
        super().__init__()
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        backbone = resnet18(weights=None)

        backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.scene_encoder = backbone

        self.feature = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim * num_atoms)
        )

    def forward(self, x):
        features = self.scene_encoder(x)
        features = self.feature(features)

        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.act_dim, self.num_atoms)

        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(logits, dim=-1)

    def reset_noise(self):
        pass

class RainbowCNNQNetwork(nn.Module):
    def __init__(self, in_channels, act_dim, num_atoms):
        super().__init__()
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        self.feature = nn.Sequential(
            nn.Linear(32 * 3 * 3, 32),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, act_dim * num_atoms)
        )

    def forward(self, x):
        features = self.cnn(x)
        features = self.feature(features)

        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.act_dim, self.num_atoms)

        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(logits, dim=-1)

    def reset_noise(self):
        pass


class RainbowCNNQNetwork_old(nn.Module):
    def __init__(self, in_channels, act_dim, num_atoms):
        super().__init__()
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        self.feature = nn.Sequential(
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim * num_atoms)
        )

    def forward(self, x):
        features = self.cnn(x)
        features = self.feature(features)

        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.act_dim, self.num_atoms)

        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(logits, dim=-1)

    def reset_noise(self):
        pass


class RainbowCNNEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.feature = nn.Sequential(
            nn.Linear(32 * 3 * 3, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.feature(x)


class RainbowDuelingDistributionalHead(nn.Module):
    def __init__(self, feature_dim, act_dim, num_atoms):
        super().__init__()
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, act_dim * num_atoms)
        )

    def forward(self, features):
        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.act_dim, self.num_atoms)

        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(logits, dim=-1)


class RainbowCNNQNetwork_autoencoder(nn.Module):
    def __init__(self, in_channels, act_dim, num_atoms, feature_dim=64):
        super().__init__()
        self.encoder = RainbowCNNEncoder(in_channels, feature_dim)
        self.q_head = RainbowDuelingDistributionalHead(feature_dim, act_dim, num_atoms)

    def forward(self, x):
        features = self.encoder(x)
        return self.q_head(features)

    def reset_noise(self):
        pass



class RainbowHybridQNetwork(nn.Module):
    def __init__(self, in_channels, scalar_dim, act_dim, num_atoms):
        super().__init__()
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        self.grid_feature = nn.Sequential(
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU()
        )

        self.scalar_feature = nn.Sequential(
            nn.Linear(scalar_dim, 16),
            nn.ReLU()
        )

        self.feature = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim * num_atoms)
        )

    def forward(self, grid_obs, scalar_obs):
        grid_x = self.cnn(grid_obs)
        grid_x = self.grid_feature(grid_x)

        scalar_x = self.scalar_feature(scalar_obs)

        x = torch.cat([grid_x, scalar_x], dim=1)
        features = self.feature(x)

        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.act_dim, self.num_atoms)

        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(logits, dim=-1)

    def reset_noise(self):
        pass



class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, transition):
        max_priority = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)

    def __len__(self):
        return len(self.buffer)
    


# Noisy Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


# light rainbow DQN
class DQN_v6:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v6.'
        self.network_name = 'Rainbow DQN Network (Version 6)'
        self.training_step = 0

        initial_obs = self.env.reset('training')  # get initial obs 
        self.obs_dim = len(initial_obs)
        self.act_dim = 5  # number of actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 5e-5
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes * 0.8
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        self.n_step = 3
        self.n_step_buffer = []

        # replay buffer parameters
        self.replay_buffer_capacity = 50000
        self.min_replay_buffer_size = 1000
        self.per_alpha = 0.6
        self.per_beta_start = 0.4
        self.per_beta_end = 1.0
        self.per_beta = self.per_beta_start
        self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, self.per_alpha)

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def build_q_network(self):
        network = DuelingQNetwork(self.obs_dim, self.act_dim)
        return network
    
    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)  # explore -> select random action
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            act = torch.argmax(q_values).item()
            return int(act)
        
    def optimize_model(self):
        beta_progress = min(1.0, self.training_step / (self.num_episodes * self.episode_steps))
        self.per_beta = self.per_beta_start + beta_progress * (self.per_beta_end - self.per_beta_start)

        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=self.per_beta)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        with torch.no_grad():
            next_q_values_online = self.q_network(next_obs_batch)
            next_actions = next_q_values_online.argmax(dim=1)

            next_q_values_target = self.target_network(next_obs_batch)
            max_target_q_values_next = next_q_values_target.gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q_values = rew_batch + (1 - done_batch) * (self.gamma ** self.n_step) * max_target_q_values_next

        q_values = self.q_network(obs_batch)
        q_values_for_actions = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)

        td_errors = target_q_values - q_values_for_actions
        losses = self.loss_fn(q_values_for_actions, target_q_values)
        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        new_priorities = td_errors.detach().abs().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        self.training_step += 1

        return loss.item()

        
    def store_transition(self, obs, act, rew, next_obs, done):
        self.n_step_buffer.append((obs, act, rew, next_obs, done))
        
        if len(self.n_step_buffer) < self.n_step and not done:
            return  
        
        while self.n_step_buffer:
            discounted_reward = 0
            final_next_obs = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]

            for i, (_, _, reward_i, next_obs_i, done_i) in enumerate(self.n_step_buffer):
                discounted_reward += (self.gamma ** i) * reward_i
                final_next_obs = next_obs_i
                final_done = done_i
                if i + 1 >= self.n_step or done_i:
                    break

            obs_0, act_0, _, _, _ = self.n_step_buffer[0]
            self.replay_buffer.add((obs_0, act_0, discounted_reward, final_next_obs, final_done))
            self.n_step_buffer.pop(0)

            if not done or len(self.n_step_buffer) < self.n_step:
                break

        if done:
            self.n_step_buffer = []

    def ready_to_update(self):
        return len(self.replay_buffer) >= self.min_replay_buffer_size
    
    def update_after_episode(self, episode):
        if episode % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        )

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)


# Full rainbow DQN
class DQN_v7:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v7.'
        self.network_name = 'Rainbow DQN Network (Version 7)'
        self.training_step = 0

        initial_obs = self.env.reset('training')  # get initial obs 
        self.obs_dim = len(initial_obs)
        self.act_dim = 5  # number of actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 5e-5
        self.target_update_freq = 10
        self.num_atoms = 51
        self.v_min = -50
        self.v_max = 300
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes * 0.8
        self.epsilon = self.epsilon_start

        self.n_step = 1
        self.n_step_buffer = []

        # replay buffer parameters
        self.replay_buffer_capacity = 50000
        self.min_replay_buffer_size = 1000
        self.per_alpha = 0.6
        self.per_beta_start = 0.4
        self.per_beta_end = 1.0
        self.per_beta = self.per_beta_start
        self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, self.per_alpha)

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def build_q_network(self):
        network = RainbowQNetwork(self.obs_dim, self.act_dim, self.num_atoms)
        return network
    
    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.q_network(obs_tensor)
            q_values = (action_probs * self.support.view(1, 1, -1)).sum(dim=2) 
        act = torch.argmax(q_values, dim=1).item()
        self.q_network.reset_noise()
        return int(act)
        
    def optimize_model(self):
        beta_progress = min(1.0, self.training_step / (self.num_episodes * self.episode_steps))
        self.per_beta = self.per_beta_start + beta_progress * (self.per_beta_end - self.per_beta_start)

        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=self.per_beta)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        batch_indices = torch.arange(self.batch_size).to(self.device)

        with torch.no_grad():
            # Double DQN: action selection with online network, value estimation with target network
            next_dist_online = self.q_network(next_obs_batch)
            next_q_online = (next_dist_online * self.support.view(1, 1, self.num_atoms)).sum(dim=2)
            next_actions = next_q_online.argmax(dim=1)

            # Distributional Bellman update
            next_dist_target = self.target_network(next_obs_batch)
            next_dist = next_dist_target[batch_indices, next_actions]

            # Bellman update
            target_support = rew_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)
            target_support = target_support.clamp(self.v_min, self.v_max)

            # Project onto fixed support
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            lower = l.clamp(0, self.num_atoms - 1)
            upper = u.clamp(0, self.num_atoms - 1)

            projected_dist = torch.zeros_like(next_dist)

            # Handle exactly on support case
            lower_weight = upper.float() - b
            upper_weight = b - lower.float()

            exact_match = (upper == lower)
            lower_weight[exact_match] = 1.0
            upper_weight[exact_match] = 0.0

            offset = (torch.arange(self.batch_size) * self.num_atoms).unsqueeze(1).to(self.device)

            projected_dist.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * lower_weight).view(-1))
            projected_dist.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * upper_weight).view(-1))

        # Get current distribution and compute loss
        dist = self.q_network(obs_batch)
        chosen_dist = dist[batch_indices, act_batch]
        chosen_dist = chosen_dist.clamp(min=1e-8)

        # cross-entropy loss between projected target distribution and current distribution
        losses = -(projected_dist * torch.log(chosen_dist)).sum(dim=1)
        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        new_priorities = losses.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        self.q_network.reset_noise()
        self.target_network.reset_noise()
        self.training_step += 1

        return loss.item()

        
    def store_transition(self, obs, act, rew, next_obs, done):
        self.n_step_buffer.append((obs, act, rew, next_obs, done))
        
        if len(self.n_step_buffer) < self.n_step and not done:
            return  
        
        while self.n_step_buffer:
            discounted_reward = 0
            final_next_obs = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]

            for i, (_, _, reward_i, next_obs_i, done_i) in enumerate(self.n_step_buffer):
                discounted_reward += (self.gamma ** i) * reward_i
                final_next_obs = next_obs_i
                final_done = done_i
                if i + 1 >= self.n_step or done_i:
                    break

            obs_0, act_0, _, _, _ = self.n_step_buffer[0]
            self.replay_buffer.add((obs_0, act_0, discounted_reward, final_next_obs, final_done))
            self.n_step_buffer.pop(0)

            if not done or len(self.n_step_buffer) < self.n_step:
                break

        if done:
            self.n_step_buffer = []

    def ready_to_update(self):
        return len(self.replay_buffer) >= self.min_replay_buffer_size
    
    def update_after_episode(self, episode):
        if episode % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        )

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)


# Full rainbow DQN for CNN and MLP (v7 like)
class DQN_v9:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v9.'
        self.network_name = 'Rainbow DQN Network (Version 9)'
        self.training_step = 0

        initial_grid_obs, initial_scalar_obs = self.env.reset('training')  # get initial obs 
        self.in_channels = initial_grid_obs.shape[0]
        self.grid_height = initial_grid_obs.shape[1]
        self.grid_width = initial_grid_obs.shape[2]
        self.scalar_dim = initial_scalar_obs.shape[0]
        self.act_dim = 5
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 5e-5
        self.target_update_freq = 10
        self.num_atoms = 51
        self.v_min = -50
        self.v_max = 300
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes * 0.8
        self.epsilon = self.epsilon_start

        self.n_step = 1
        self.n_step_buffer = []

        # replay buffer parameters
        self.replay_buffer_capacity = 50000
        self.min_replay_buffer_size = 1000
        self.per_alpha = 0.6
        self.per_beta_start = 0.4
        self.per_beta_end = 1.0
        self.per_beta = self.per_beta_start
        self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, self.per_alpha)

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def build_q_network(self):
        network = RainbowHybridQNetwork(self.in_channels, self.scalar_dim, self.act_dim, self.num_atoms)
        return network
    
    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        
        grid_obs, scalar_obs = obs

        grid_tensor = torch.from_numpy(grid_obs).float().unsqueeze(0).to(self.device)
        scalar_tensor = torch.from_numpy(scalar_obs).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.q_network(grid_tensor, scalar_tensor)
            q_values = (action_probs * self.support.view(1, 1, -1)).sum(dim=2) 
        act = torch.argmax(q_values, dim=1).item()
        self.q_network.reset_noise()
        return int(act)
        
    def optimize_model(self):
        beta_progress = min(1.0, self.training_step / (self.num_episodes * self.episode_steps))
        self.per_beta = self.per_beta_start + beta_progress * (self.per_beta_end - self.per_beta_start)

        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=self.per_beta)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        grid_batch = torch.from_numpy(np.stack([obs[0] for obs in obs_batch])).float().to(self.device)

        scalar_batch = torch.from_numpy(np.stack([obs[1] for obs in obs_batch])).float().to(self.device)

        next_grid_batch = torch.from_numpy(np.stack([obs[0] for obs in next_obs_batch])).float().to(self.device)

        next_scalar_batch = torch.from_numpy(np.stack([obs[1] for obs in next_obs_batch])).float().to(self.device)

        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        batch_indices = torch.arange(self.batch_size).to(self.device)

        with torch.no_grad():
            # Double DQN: action selection with online network, value estimation with target network
            next_dist_online = self.q_network(next_grid_batch, next_scalar_batch)
            next_q_online = (next_dist_online * self.support.view(1, 1, self.num_atoms)).sum(dim=2)
            next_actions = next_q_online.argmax(dim=1)

            # Distributional Bellman update
            next_dist_target = self.target_network(next_grid_batch, next_scalar_batch)
            next_dist = next_dist_target[batch_indices, next_actions]

            # Bellman update
            target_support = rew_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)
            target_support = target_support.clamp(self.v_min, self.v_max)

            # Project onto fixed support
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            lower = l.clamp(0, self.num_atoms - 1)
            upper = u.clamp(0, self.num_atoms - 1)

            projected_dist = torch.zeros_like(next_dist)

            # Handle exactly on support case
            lower_weight = upper.float() - b
            upper_weight = b - lower.float()

            exact_match = (upper == lower)
            lower_weight[exact_match] = 1.0
            upper_weight[exact_match] = 0.0

            offset = (torch.arange(self.batch_size) * self.num_atoms).unsqueeze(1).to(self.device)

            projected_dist.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * lower_weight).view(-1))
            projected_dist.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * upper_weight).view(-1))

        # Get current distribution and compute loss
        dist = self.q_network(grid_batch, scalar_batch)
        chosen_dist = dist[batch_indices, act_batch]
        chosen_dist = chosen_dist.clamp(min=1e-8)

        # cross-entropy loss between projected target distribution and current distribution
        losses = -(projected_dist * torch.log(chosen_dist)).sum(dim=1)
        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        new_priorities = losses.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        self.q_network.reset_noise()
        self.target_network.reset_noise()
        self.training_step += 1

        return loss.item()

        
    def store_transition(self, obs, act, rew, next_obs, done):
        self.n_step_buffer.append((obs, act, rew, next_obs, done))
        
        if len(self.n_step_buffer) < self.n_step and not done:
            return  
        
        while self.n_step_buffer:
            discounted_reward = 0
            final_next_obs = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]

            for i, (_, _, reward_i, next_obs_i, done_i) in enumerate(self.n_step_buffer):
                discounted_reward += (self.gamma ** i) * reward_i
                final_next_obs = next_obs_i
                final_done = done_i
                if i + 1 >= self.n_step or done_i:
                    break

            obs_0, act_0, _, _, _ = self.n_step_buffer[0]
            self.replay_buffer.add((obs_0, act_0, discounted_reward, final_next_obs, final_done))
            self.n_step_buffer.pop(0)

            if not done or len(self.n_step_buffer) < self.n_step:
                break

        if done:
            self.n_step_buffer = []

    def ready_to_update(self):
        return len(self.replay_buffer) >= self.min_replay_buffer_size
    
    def update_after_episode(self, episode):
        if episode % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        )

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)



# Full rainbow DQN for CNN (v7 like)
class DQN_v8:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v8.'
        self.network_name = 'Rainbow DQN Network (Version 8)'
        self.training_step = 0

        initial_obs = self.env.reset('training')  # get initial obs 
        self.in_channels = initial_obs.shape[0]
        self.grid_height = initial_obs.shape[1]
        self.grid_width = initial_obs.shape[2]
        self.act_dim = 5
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 5e-5
        self.target_update_freq = 10
        self.num_atoms = 51
        self.v_min = -50
        self.v_max = 300
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        # self.epsilon_decay_steps = self.num_episodes * 0.8
        self.epsilon_decay_steps = 8000
        self.epsilon = self.epsilon_start

        self.n_step = 1
        self.n_step_buffer = []

        # replay buffer parameters
        self.replay_buffer_capacity = 50000
        self.min_replay_buffer_size = 1000
        self.per_alpha = 0.6
        self.per_beta_start = 0.4
        self.per_beta_end = 1.0
        self.per_beta = self.per_beta_start
        self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, self.per_alpha)

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def build_q_network(self):
        network = RainbowCNNQNetwork(self.in_channels, self.act_dim, self.num_atoms)
        return network
    
    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.q_network(obs_tensor)
            q_values = (action_probs * self.support.view(1, 1, -1)).sum(dim=2) 
        act = torch.argmax(q_values, dim=1).item()
        self.q_network.reset_noise()
        return int(act)
        
    def optimize_model(self):
        beta_progress = min(1.0, self.training_step / (self.num_episodes * self.episode_steps))
        self.per_beta = self.per_beta_start + beta_progress * (self.per_beta_end - self.per_beta_start)

        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=self.per_beta)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.from_numpy(np.stack(next_obs_batch)).float().to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        batch_indices = torch.arange(self.batch_size).to(self.device)

        with torch.no_grad():
            # Double DQN: action selection with online network, value estimation with target network
            next_dist_online = self.q_network(next_obs_batch)
            next_q_online = (next_dist_online * self.support.view(1, 1, self.num_atoms)).sum(dim=2)
            next_actions = next_q_online.argmax(dim=1)

            # Distributional Bellman update
            next_dist_target = self.target_network(next_obs_batch)
            next_dist = next_dist_target[batch_indices, next_actions]

            # Bellman update
            target_support = rew_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)
            target_support = target_support.clamp(self.v_min, self.v_max)

            # Project onto fixed support
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            lower = l.clamp(0, self.num_atoms - 1)
            upper = u.clamp(0, self.num_atoms - 1)

            projected_dist = torch.zeros_like(next_dist)

            # Handle exactly on support case
            lower_weight = upper.float() - b
            upper_weight = b - lower.float()

            exact_match = (upper == lower)
            lower_weight[exact_match] = 1.0
            upper_weight[exact_match] = 0.0

            offset = (torch.arange(self.batch_size) * self.num_atoms).unsqueeze(1).to(self.device)

            projected_dist.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * lower_weight).view(-1))
            projected_dist.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * upper_weight).view(-1))

        # Get current distribution and compute loss
        dist = self.q_network(obs_batch)
        chosen_dist = dist[batch_indices, act_batch]
        chosen_dist = chosen_dist.clamp(min=1e-8)

        # cross-entropy loss between projected target distribution and current distribution
        losses = -(projected_dist * torch.log(chosen_dist)).sum(dim=1)
        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        new_priorities = losses.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        self.q_network.reset_noise()
        self.target_network.reset_noise()
        self.training_step += 1

        return loss.item()

        
    def store_transition(self, obs, act, rew, next_obs, done):
        self.n_step_buffer.append((obs, act, rew, next_obs, done))
        
        if len(self.n_step_buffer) < self.n_step and not done:
            return  
        
        while self.n_step_buffer:
            discounted_reward = 0
            final_next_obs = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]

            for i, (_, _, reward_i, next_obs_i, done_i) in enumerate(self.n_step_buffer):
                discounted_reward += (self.gamma ** i) * reward_i
                final_next_obs = next_obs_i
                final_done = done_i
                if i + 1 >= self.n_step or done_i:
                    break

            obs_0, act_0, _, _, _ = self.n_step_buffer[0]
            self.replay_buffer.add((obs_0, act_0, discounted_reward, final_next_obs, final_done))
            self.n_step_buffer.pop(0)

            if not done or len(self.n_step_buffer) < self.n_step:
                break

        if done:
            self.n_step_buffer = []

    def ready_to_update(self):
        return len(self.replay_buffer) >= self.min_replay_buffer_size
    
    def update_after_episode(self, episode):
        if episode % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        )

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)


class DQN_v8_old(DQN_v8):
    def build_q_network(self):
        network = RainbowCNNQNetwork_old(self.in_channels, self.act_dim, self.num_atoms)
        return network



# Full Rainbow DQN for Resnet18
class DQN_v10:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v10.'
        self.network_name = 'Rainbow DQN Network (Version 10)'
        self.training_step = 0

        initial_obs = self.env.reset('training')  # get initial obs 
        self.in_channels = initial_obs.shape[0]
        self.grid_height = initial_obs.shape[1]
        self.grid_width = initial_obs.shape[2]
        self.act_dim = 5
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 5e-5
        self.target_update_freq = 10
        self.num_atoms = 51
        self.v_min = -50
        self.v_max = 300
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes * 0.8
        self.epsilon = self.epsilon_start

        self.n_step = 1
        self.n_step_buffer = []

        # replay buffer parameters
        self.replay_buffer_capacity = 50000
        self.min_replay_buffer_size = 1000
        self.per_alpha = 0.6
        self.per_beta_start = 0.4
        self.per_beta_end = 1.0
        self.per_beta = self.per_beta_start
        self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, self.per_alpha)

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def build_q_network(self):
        network = RainbowResNet18QNetwork(self.in_channels, self.act_dim, self.num_atoms)
        return network
    
    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        was_training = self.q_network.training
        self.q_network.eval()
        try:
            with torch.no_grad():
                action_probs = self.q_network(obs_tensor)
                q_values = (action_probs * self.support.view(1, 1, -1)).sum(dim=2)
        finally:
            if was_training:
                self.q_network.train()

        act = torch.argmax(q_values, dim=1).item()
        self.q_network.reset_noise()
        return int(act)
        
    def optimize_model(self):
        beta_progress = min(1.0, self.training_step / (self.num_episodes * self.episode_steps))
        self.per_beta = self.per_beta_start + beta_progress * (self.per_beta_end - self.per_beta_start)

        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=self.per_beta)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.from_numpy(np.stack(next_obs_batch)).float().to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        batch_indices = torch.arange(self.batch_size).to(self.device)

        with torch.no_grad():
            # Double DQN: action selection with online network, value estimation with target network
            next_dist_online = self.q_network(next_obs_batch)
            next_q_online = (next_dist_online * self.support.view(1, 1, self.num_atoms)).sum(dim=2)
            next_actions = next_q_online.argmax(dim=1)

            # Distributional Bellman update
            next_dist_target = self.target_network(next_obs_batch)
            next_dist = next_dist_target[batch_indices, next_actions]

            # Bellman update
            target_support = rew_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)
            target_support = target_support.clamp(self.v_min, self.v_max)

            # Project onto fixed support
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            lower = l.clamp(0, self.num_atoms - 1)
            upper = u.clamp(0, self.num_atoms - 1)

            projected_dist = torch.zeros_like(next_dist)

            # Handle exactly on support case
            lower_weight = upper.float() - b
            upper_weight = b - lower.float()

            exact_match = (upper == lower)
            lower_weight[exact_match] = 1.0
            upper_weight[exact_match] = 0.0

            offset = (torch.arange(self.batch_size) * self.num_atoms).unsqueeze(1).to(self.device)

            projected_dist.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * lower_weight).view(-1))
            projected_dist.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * upper_weight).view(-1))

        # Get current distribution and compute loss
        dist = self.q_network(obs_batch)
        chosen_dist = dist[batch_indices, act_batch]
        chosen_dist = chosen_dist.clamp(min=1e-8)

        # cross-entropy loss between projected target distribution and current distribution
        losses = -(projected_dist * torch.log(chosen_dist)).sum(dim=1)
        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        new_priorities = losses.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        self.q_network.reset_noise()
        self.target_network.reset_noise()
        self.training_step += 1

        return loss.item()

        
    def store_transition(self, obs, act, rew, next_obs, done):
        self.n_step_buffer.append((obs, act, rew, next_obs, done))
        
        if len(self.n_step_buffer) < self.n_step and not done:
            return  
        
        while self.n_step_buffer:
            discounted_reward = 0
            final_next_obs = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]

            for i, (_, _, reward_i, next_obs_i, done_i) in enumerate(self.n_step_buffer):
                discounted_reward += (self.gamma ** i) * reward_i
                final_next_obs = next_obs_i
                final_done = done_i
                if i + 1 >= self.n_step or done_i:
                    break

            obs_0, act_0, _, _, _ = self.n_step_buffer[0]
            self.replay_buffer.add((obs_0, act_0, discounted_reward, final_next_obs, final_done))
            self.n_step_buffer.pop(0)

            if not done or len(self.n_step_buffer) < self.n_step:
                break

        if done:
            self.n_step_buffer = []

    def ready_to_update(self):
        return len(self.replay_buffer) >= self.min_replay_buffer_size
    
    def update_after_episode(self, episode):
        if episode % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        )

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)

# Full rainbow DQN for CNN using separated encoder and Q head
class DQN_v11:
    def __init__(self, env, encoder_path=None, freeze_encoder=False, encoder_type="cnn"):
        if freeze_encoder and encoder_path is None:
            raise ValueError("freeze_encoder=True requires a pretrained encoder_path.")

        encoder_type = encoder_type.lower()
        valid_encoder_types = {"cnn", "lenet5", "alexnet8", "resnet18"}
        if encoder_type not in valid_encoder_types:
            raise ValueError(
                f"encoder_type must be one of {sorted(valid_encoder_types)}, got {encoder_type!r}"
            )

        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v11.'
        self.network_name = 'Rainbow DQN Network (Version 11)'
        self.training_step = 0
        self.encoder_path = encoder_path
        self.freeze_encoder = freeze_encoder
        self.encoder_type = encoder_type

        initial_obs = self.env.reset('training')  # get initial obs 
        self.in_channels = initial_obs.shape[0]
        self.grid_height = initial_obs.shape[1]
        self.grid_width = initial_obs.shape[2]
        self.act_dim = 5
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 5e-5
        self.target_update_freq = 10
        self.num_atoms = 51
        self.feature_dim = 64
        self.v_min = -50
        self.v_max = 300
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes * 0.8
        self.epsilon = self.epsilon_start

        self.n_step = 1
        self.n_step_buffer = []

        # replay buffer parameters
        self.replay_buffer_capacity = 50000
        self.min_replay_buffer_size = 1000
        self.per_alpha = 0.6
        self.per_beta_start = 0.4
        self.per_beta_end = 1.0
        self.per_beta = self.per_beta_start
        self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, self.per_alpha)

        self.q_network = self.build_q_network().to(self.device)
        if self.encoder_path is not None:
            self.load_encoder(self.encoder_path)
            print(f'Loaded encoder from: {self.encoder_path}')
        if self.freeze_encoder:
            for parameter in self.q_network.encoder.parameters():
                parameter.requires_grad = False
            print('Encoder frozen: only the Q head will be trained')
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        trainable_parameters = [
            parameter
            for parameter in self.q_network.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = optim.Adam(trainable_parameters, lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def build_q_network(self):
        network_classes = {
            "cnn": RainbowCNNQNetwork_autoencoder,
            "lenet5": RainbowLeNet5QNetwork,
            "alexnet8": RainbowAlexNet8QNetwork,
            "resnet18": PretrainedRainbowResNet18QNetwork,
        }
        return network_classes[self.encoder_type](
            self.in_channels,
            self.act_dim,
            self.num_atoms,
            feature_dim=self.feature_dim,
        )

    def load_encoder(self, encoder_path):
        checkpoint = torch.load(encoder_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
            checkpoint_encoder_type = checkpoint.get("encoder_type")
            checkpoint_feature_dim = checkpoint.get("feature_dim")
            checkpoint_in_channels = checkpoint.get("in_channels")
            if checkpoint_encoder_type is not None and checkpoint_encoder_type != self.encoder_type:
                raise ValueError(
                    f"Encoder type mismatch: checkpoint has {checkpoint_encoder_type!r}, "
                    f"but DQN_v11 was built with {self.encoder_type!r}."
                )
            if checkpoint_feature_dim is not None and checkpoint_feature_dim != self.feature_dim:
                raise ValueError(
                    f"Encoder feature_dim mismatch: checkpoint has {checkpoint_feature_dim}, "
                    f"but DQN_v11 expects {self.feature_dim}."
                )
            if checkpoint_in_channels is not None and checkpoint_in_channels != self.in_channels:
                raise ValueError(
                    f"Encoder input channel mismatch: checkpoint has {checkpoint_in_channels}, "
                    f"but env observations have {self.in_channels} channels."
                )
            encoder_state_dict = checkpoint["encoder_state_dict"]
        else:
            encoder_state_dict = checkpoint
        self.q_network.encoder.load_state_dict(encoder_state_dict)
    
    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        was_training = self.q_network.training
        self.q_network.eval()
        try:
            with torch.no_grad():
                action_probs = self.q_network(obs_tensor)
                q_values = (action_probs * self.support.view(1, 1, -1)).sum(dim=2)
        finally:
            if was_training:
                self.q_network.train()
        act = torch.argmax(q_values, dim=1).item()
        self.q_network.reset_noise()
        return int(act)
        
    def optimize_model(self):
        beta_progress = min(1.0, self.training_step / (self.num_episodes * self.episode_steps))
        self.per_beta = self.per_beta_start + beta_progress * (self.per_beta_end - self.per_beta_start)

        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=self.per_beta)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.from_numpy(np.stack(next_obs_batch)).float().to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        batch_indices = torch.arange(self.batch_size).to(self.device)

        with torch.no_grad():
            # Double DQN: action selection with online network, value estimation with target network
            next_dist_online = self.q_network(next_obs_batch)
            next_q_online = (next_dist_online * self.support.view(1, 1, self.num_atoms)).sum(dim=2)
            next_actions = next_q_online.argmax(dim=1)

            # Distributional Bellman update
            next_dist_target = self.target_network(next_obs_batch)
            next_dist = next_dist_target[batch_indices, next_actions]

            # Bellman update
            target_support = rew_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)
            target_support = target_support.clamp(self.v_min, self.v_max)

            # Project onto fixed support
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            lower = l.clamp(0, self.num_atoms - 1)
            upper = u.clamp(0, self.num_atoms - 1)

            projected_dist = torch.zeros_like(next_dist)

            # Handle exactly on support case
            lower_weight = upper.float() - b
            upper_weight = b - lower.float()

            exact_match = (upper == lower)
            lower_weight[exact_match] = 1.0
            upper_weight[exact_match] = 0.0

            offset = (torch.arange(self.batch_size) * self.num_atoms).unsqueeze(1).to(self.device)

            projected_dist.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * lower_weight).view(-1))
            projected_dist.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * upper_weight).view(-1))

        # Get current distribution and compute loss
        dist = self.q_network(obs_batch)
        chosen_dist = dist[batch_indices, act_batch]
        chosen_dist = chosen_dist.clamp(min=1e-8)

        # cross-entropy loss between projected target distribution and current distribution
        losses = -(projected_dist * torch.log(chosen_dist)).sum(dim=1)
        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        new_priorities = losses.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        self.q_network.reset_noise()
        self.target_network.reset_noise()
        self.training_step += 1

        return loss.item()

        
    def store_transition(self, obs, act, rew, next_obs, done):
        self.n_step_buffer.append((obs, act, rew, next_obs, done))
        
        if len(self.n_step_buffer) < self.n_step and not done:
            return  
        
        while self.n_step_buffer:
            discounted_reward = 0
            final_next_obs = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]

            for i, (_, _, reward_i, next_obs_i, done_i) in enumerate(self.n_step_buffer):
                discounted_reward += (self.gamma ** i) * reward_i
                final_next_obs = next_obs_i
                final_done = done_i
                if i + 1 >= self.n_step or done_i:
                    break

            obs_0, act_0, _, _, _ = self.n_step_buffer[0]
            self.replay_buffer.add((obs_0, act_0, discounted_reward, final_next_obs, final_done))
            self.n_step_buffer.pop(0)

            if not done or len(self.n_step_buffer) < self.n_step:
                break

        if done:
            self.n_step_buffer = []

    def ready_to_update(self):
        return len(self.replay_buffer) >= self.min_replay_buffer_size
    
    def update_after_episode(self, episode):
        if episode % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        )

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)
