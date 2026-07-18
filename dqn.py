import random
import numpy as np
# TensorFlow is only required by the legacy DQN_v0 demo. Keep it disabled so
# importing this module does not load TensorFlow alongside PyTorch.
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim


class HybridQNetwork(nn.Module):
    """Small CNN + scalar MLP that directly predicts one Q value per action."""

    def __init__(self, in_channels, grid_height, grid_width, scalar_dim, act_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        grid_feature_dim = 16 * grid_height * grid_width
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 16),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(grid_feature_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, grid_obs, scalar_obs):
        grid_features = self.cnn(grid_obs)
        scalar_features = self.scalar_mlp(scalar_obs)
        return self.q_head(torch.cat((grid_features, scalar_features), dim=1))


class DQN_hybrid:
    """Vanilla one-step DQN for environments returning (grid, scalar) observations."""

    def __init__(self, env):
        self.env = env
        self.variant = env.variant
        self.data_dir = env.data_dir
        self.file_name = 'DQN_hybrid.'
        self.network_name = 'Vanilla Hybrid DQN'

        initial_grid_obs, initial_scalar_obs = env.reset('training')
        self.in_channels, self.grid_height, self.grid_width = initial_grid_obs.shape
        self.scalar_dim = initial_scalar_obs.shape[0]
        self.act_dim = 5

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        self.episode_steps = env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 1e-4
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes * 0.8
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        self.replay_buffer_capacity = 50000
        self.min_replay_buffer_size = 1000
        self.replay_buffer = []

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def build_q_network(self):
        return HybridQNetwork(
            self.in_channels,
            self.grid_height,
            self.grid_width,
            self.scalar_dim,
            self.act_dim,
        )

    def _obs_to_tensors(self, observations):
        grid = torch.from_numpy(np.stack([obs[0] for obs in observations])).float().to(self.device)
        scalar = torch.from_numpy(np.stack([obs[1] for obs in observations])).float().to(self.device)
        return grid, scalar

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.act_dim))

        grid_obs, scalar_obs = obs
        grid_tensor = torch.from_numpy(grid_obs).float().unsqueeze(0).to(self.device)
        scalar_tensor = torch.from_numpy(scalar_obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(grid_tensor, scalar_tensor)
        return int(q_values.argmax(dim=1).item())

    def optimize_model(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)
        grid_batch, scalar_batch = self._obs_to_tensors(obs_batch)
        next_grid_batch, next_scalar_batch = self._obs_to_tensors(next_obs_batch)
        actions = torch.tensor(act_batch, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rew_batch, dtype=torch.float32, device=self.device)
        dones = torch.tensor(done_batch, dtype=torch.float32, device=self.device)

        q_values = self.q_network(grid_batch, scalar_batch).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_grid_batch, next_scalar_batch).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def store_transition(self, obs, act, rew, next_obs, done):
        if len(self.replay_buffer) >= self.replay_buffer_capacity:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((obs, act, rew, next_obs, done))

    def ready_to_update(self):
        return len(self.replay_buffer) >= self.min_replay_buffer_size

    def update_after_episode(self, episode):
        if (episode + 1) % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps,
        )

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)

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


# DQN Demo with Tensorflow, can be used to check CPU training and set up
class DQN_v0:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v0.'
        self.network_name = 'DQN Network (Version 0)'

        initial_obs = self.env.reset('training')  # get initial obs 
        self.obs_dim = len(initial_obs)
        self.act_dim = 5  # number of actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

        # Hyperparameters
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 50
        self.batch_size = 32
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.epsilon_start = 0.5
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # replay buffer parameters
        self.replay_buffer_capacity = 10000
        self.min_replay_buffer_size = 500
        self.replay_buffer = []  # initialize replay buffer as empty list

        # optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # build Q-network
        self.q_network = self.build_q_network()
        self.target_network = self.build_q_network()
        self.target_network.set_weights(self.q_network.get_weights())

    def build_q_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.act_dim)
        ])
        return model

    # action selector
    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)  # explore -> select random action
        else:
            q_values = self.q_network(np.array([obs]))  # get Q-values from the network
            act = np.argmax(q_values[0])  # select action with highest Q-value
            return int(act) 
        
    # optimization step
    def optimize_model(self):

        # sample a batch of transitions from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = np.array(obs_batch)
        act_batch = np.array(act_batch)
        rew_batch = np.array(rew_batch)
        next_obs_batch = np.array(next_obs_batch)
        done_batch = np.array(done_batch)

        # compute target Q-values using the target network
        # y = r + gamma * (1-d) * max_a' Q_target(s', a')
        target_q_values_next = self.target_network(next_obs_batch)
        max_target_q_values_next = np.max(target_q_values_next, axis=1)
        target_q_values = rew_batch + (1 - done_batch) * self.gamma * max_target_q_values_next

        with tf.GradientTape() as tape:
            q_values = self.q_network(obs_batch)
            action_masks = tf.one_hot(act_batch, self.act_dim)
            q_values_for_actions = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = self.loss_fn(target_q_values, q_values_for_actions)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        return loss.numpy()
    
    # function to store transition in replay buffer
    def store_transition(self, obs, act, rew, next_obs, done):
        if len(self.replay_buffer) >= self.replay_buffer_capacity:
            self.replay_buffer.pop(0)

        self.replay_buffer.append((obs, act, rew, next_obs, done))

    def ready_to_update(self):
        return len(self.replay_buffer) >= self.min_replay_buffer_size

    def update_after_episode(self, episode):
        if episode % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        )

    def save(self, model_path):
        self.q_network.save(model_path)




# DQN network with pytorch baseline
class DQN_v1:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v1.'
        self.network_name = 'DQN Network (Version 1)'

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
        self.gamma = 0.8
        self.learning_rate = 5e-5
        self.epsilon_start = 0.5
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes / 2
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # replay buffer parameters
        self.replay_buffer_capacity = 10000
        self.min_replay_buffer_size = 500
        self.replay_buffer = []  # initialize replay buffer as empty list

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_q_network(self):
        network = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.act_dim)
        )
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
        # sample a batch of transitions from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # compute target Q-values using the target network
        target_q_values_next = self.target_network(next_obs_batch)
        max_target_q_values_next = target_q_values_next.max(dim=1)[0]
        target_q_values = rew_batch + (1 - done_batch) * self.gamma * max_target_q_values_next

        q_values = self.q_network(obs_batch)
        action_masks = torch.nn.functional.one_hot(act_batch, num_classes=self.act_dim)
        q_values_for_actions = (q_values * action_masks).sum(dim=1)

        loss = self.loss_fn(q_values_for_actions, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def store_transition(self, obs, act, rew, next_obs, done):
        if len(self.replay_buffer) >= self.replay_buffer_capacity:
            self.replay_buffer.pop(0)

        self.replay_buffer.append((obs, act, rew, next_obs, done))

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
    
  
# agressive DQN
class DQN_v2:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v2.'
        self.network_name = 'DQN Network (Version 2)'

        initial_obs = self.env.reset('training')  # get initial obs 
        self.obs_dim = len(initial_obs)
        self.act_dim = 5  # number of actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 5000
        self.batch_size = 256
        self.gamma = 0.99
        self.learning_rate = 3e-4
        self.epsilon_start = 0.9
        self.epsilon_end = 0.02
        self.epsilon_decay_steps = int(self.num_episodes * 0.7)
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # replay buffer parameters
        self.replay_buffer_capacity = 100000
        self.min_replay_buffer_size = 5000
        self.replay_buffer = []  # initialize replay buffer as empty list

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.learning_rate, weight_decay=1e-4) # L2 regularization
        self.loss_fn = nn.SmoothL1Loss()

    def build_q_network(self):
        network = nn.Sequential(
            nn.Linear(self.obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.act_dim)
        )
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
        # sample a batch of transitions from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # compute target Q-values using the target network
        target_q_values_next = self.target_network(next_obs_batch)
        max_target_q_values_next = target_q_values_next.max(dim=1)[0]
        target_q_values = rew_batch + (1 - done_batch) * self.gamma * max_target_q_values_next

        q_values = self.q_network(obs_batch)
        action_masks = torch.nn.functional.one_hot(act_batch, num_classes=self.act_dim)
        q_values_for_actions = (q_values * action_masks).sum(dim=1)

        loss = self.loss_fn(q_values_for_actions, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()
    
    def store_transition(self, obs, act, rew, next_obs, done):
        if len(self.replay_buffer) >= self.replay_buffer_capacity:
            self.replay_buffer.pop(0)

        self.replay_buffer.append((obs, act, rew, next_obs, done))

    def ready_to_update(self):
        return len(self.replay_buffer) >= self.min_replay_buffer_size
    
    def update_after_episode(self, episode):
        if episode % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        self.epsilon = max(self.epsilon_end, self.epsilon - decay)

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)



# DQN network with deeper network
class DQN_v3:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v3.'
        self.network_name = 'DQN Network (Version 3)'

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
        self.gamma = 0.8
        self.learning_rate = 5e-5
        self.epsilon_start = 0.5
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes / 2
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # replay buffer parameters
        self.replay_buffer_capacity = 10000
        self.min_replay_buffer_size = 500
        self.replay_buffer = []  # initialize replay buffer as empty list

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_q_network(self):
        network = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.act_dim)
        )
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
        # sample a batch of transitions from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # compute target Q-values using the target network
        target_q_values_next = self.target_network(next_obs_batch)
        max_target_q_values_next = target_q_values_next.max(dim=1)[0]
        target_q_values = rew_batch + (1 - done_batch) * self.gamma * max_target_q_values_next

        q_values = self.q_network(obs_batch)
        action_masks = torch.nn.functional.one_hot(act_batch, num_classes=self.act_dim)
        q_values_for_actions = (q_values * action_masks).sum(dim=1)

        loss = self.loss_fn(q_values_for_actions, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def store_transition(self, obs, act, rew, next_obs, done):
        if len(self.replay_buffer) >= self.replay_buffer_capacity:
            self.replay_buffer.pop(0)

        self.replay_buffer.append((obs, act, rew, next_obs, done))

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




# Dueling DQN
class DQN_v4:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v4.'
        self.network_name = 'Dueling DQN Network (Version 4)'

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
        self.gamma = 0.9
        self.learning_rate = 5e-5
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes * 0.8
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # replay buffer parameters
        self.replay_buffer_capacity = 10000
        self.min_replay_buffer_size = 500
        self.replay_buffer = []  # initialize replay buffer as empty list

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

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
        # sample a batch of transitions from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # compute target Q-values using the target network
        target_q_values_next = self.target_network(next_obs_batch)
        max_target_q_values_next = target_q_values_next.max(dim=1)[0]
        target_q_values = rew_batch + (1 - done_batch) * self.gamma * max_target_q_values_next

        q_values = self.q_network(obs_batch)
        action_masks = torch.nn.functional.one_hot(act_batch, num_classes=self.act_dim)
        q_values_for_actions = (q_values * action_masks).sum(dim=1)

        loss = self.loss_fn(q_values_for_actions, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def store_transition(self, obs, act, rew, next_obs, done):
        if len(self.replay_buffer) >= self.replay_buffer_capacity:
            self.replay_buffer.pop(0)

        self.replay_buffer.append((obs, act, rew, next_obs, done))

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



# Double Dueling DQN
class DQN_v5:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v5.'
        self.network_name = 'Double Dueling DQN Network (Version 5)'

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
        self.gamma = 0.8
        self.learning_rate = 5e-5
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes * 0.8
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # replay buffer parameters
        self.replay_buffer_capacity = 10000
        self.min_replay_buffer_size = 500
        self.replay_buffer = []  # initialize replay buffer as empty list

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

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
        # sample a batch of transitions from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Double DQN target
        with torch.no_grad():
            next_q_values_online = self.q_network(next_obs_batch)
            next_actions = next_q_values_online.argmax(dim=1)

            next_q_values_target = self.target_network(next_obs_batch)
            max_target_q_values_next = next_q_values_target.gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q_values = rew_batch + (1 - done_batch) * self.gamma * max_target_q_values_next

        q_values = self.q_network(obs_batch)
        action_masks = torch.nn.functional.one_hot(act_batch, num_classes=self.act_dim)
        q_values_for_actions = (q_values * action_masks).sum(dim=1)

        loss = self.loss_fn(q_values_for_actions, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def store_transition(self, obs, act, rew, next_obs, done):
        if len(self.replay_buffer) >= self.replay_buffer_capacity:
            self.replay_buffer.pop(0)

        self.replay_buffer.append((obs, act, rew, next_obs, done))

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
