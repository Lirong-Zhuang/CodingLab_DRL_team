import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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

class rainbow_dqn:
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
