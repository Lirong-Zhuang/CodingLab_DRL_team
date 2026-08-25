import random
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

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

        # how good is to be in this state, regardless of the action
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # how much better is one action compared to the others in this state
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
        # the action taken
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


class DQN_CNN_v1:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_CNN_v1.'
        self.network_name = 'DQN Convolutional Network (Version 1)'

        # 1. Get initial observation shape dynamically
        initial_obs = self.env.reset('training')  # Returns shape (4, 5, 5) from get_cnn_obs()
        self.in_channels = initial_obs.shape[0]  # 4 channels
        self.grid_size = initial_obs.shape[1]  # 5x5 grid
        self.act_dim = 5  # 0: nothing, 1: up, 2: right, 3: down, 4: left

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters (Kept identical to your teammate's setup)
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 5e-5
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 4000
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # Replay buffer parameters
        self.replay_buffer_capacity = 20000
        self.min_replay_buffer_size = 2000
        self.replay_buffer = []

        # Build networks
        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_q_network(self):
        # Sweeping 5x5 map space using customized 16 -> 32 filters layout
        network = nn.Sequential(
            # Layer 1: Input (4, 5, 5) -> Output (16, 5, 5)
            nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: Input (16, 5, 5) -> Output (32, 3, 3)  [Valid padding shrinks spatial dimensions to 3x3]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Flatten Layer: Converts (32, 3, 3) feature map to a flat 288 vector
            nn.Flatten(),
            # Layer 3: Decision Dense Layer
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU(),
            # Layer 4: Output Layer maps to action dimensions
            nn.Linear(64, self.act_dim)
        )
        return network

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        else:
            # FIX 1: Convert the 3D NumPy array cleanly to a tensor and add batch dimension [1, 4, 5, 5]
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            act = torch.argmax(q_values).item()
            return int(act)

    def optimize_model(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        # FIX 2: Use np.stack() to pack a list of independent 3D arrays into solid blocks of shapes [64, 4, 5, 5]
        obs_batch = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.from_numpy(np.stack(next_obs_batch)).float().to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Compute targets
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
        # Raw NumPy arrays from your get_cnn_obs() flow into the list safely
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

class DQN_CNN_v2:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_CNN_v2.'
        self.network_name = 'DQN Convolutional Network (Version 2)'

        # 1. Get initial observation shape dynamically
        initial_obs = self.env.reset('training')  # Returns shape (6, 5, 5) from get_cnn_obs()
        self.in_channels = initial_obs.shape[0]  # 6 channels
        self.grid_size = initial_obs.shape[1]  # 5x5 grid
        self.act_dim = 5  # 0: nothing, 1: up, 2: right, 3: down, 4: left

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters (Kept identical to your teammate's setup)
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 5e-5
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 4000
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # Replay buffer parameters
        self.replay_buffer_capacity = 20000
        self.min_replay_buffer_size = 2000
        self.replay_buffer = []

        # Build networks
        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_q_network(self):
        # Sweeping 5x5 map space using customized 16 -> 32 filters layout
        network = nn.Sequential(
            # Layer 1: Input (x, 5, 5) -> Output (16, 5, 5)
            nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: Input (16, 5, 5) -> Output (32, 3, 3)  [Valid padding shrinks spatial dimensions to 3x3]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Flatten Layer: Converts (32, 3, 3) feature map to a flat 288 vector
            nn.Flatten(),
            # Layer 3: Decision Dense Layer
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            # Layer 4: Output Layer maps to action dimensions
            nn.Linear(128, self.act_dim)
        )
        return network

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        else:
            # FIX 1: Convert the 3D NumPy array cleanly to a tensor and add batch dimension [1, 6, 5, 5]
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            act = torch.argmax(q_values).item()
            return int(act)

    def optimize_model(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        # FIX 2: Use np.stack() to pack a list of independent 3D arrays into solid blocks of shapes [128, 6, 5, 5]
        obs_batch = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.from_numpy(np.stack(next_obs_batch)).float().to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        with torch.no_grad():
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
        # Raw NumPy arrays from your get_cnn_obs() flow into the list safely
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

class DuelingQNetwork_CNN(nn.Module):
    def __init__(self, in_channels, act_dim):
        super().__init__()
        self.feature = nn.Sequential(
            # Layer 1: Input (6, 5, 5) -> Output (16, 5, 5)
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: Input (16, 5, 5) -> Output (32, 3, 3)  [Valid padding shrinks spatial dimensions to 3x3]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Flatten Layer: Converts (32, 3, 3) feature map to a flat 288 vector
            nn.Flatten()
        )

        # how good is to be in this state, regardless of the action
        self.value_stream = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # how much better is one action compared to the others in this state
        self.advantage_stream = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# dueling DQN with CNN
class DQN_CNN_v3:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_CNN_v3.'
        self.network_name = 'DQN Convolutional Network (Version 3)'

        # 1. Get initial observation shape dynamically
        initial_obs = self.env.reset('training')  # Returns shape (6, 5, 5) from get_cnn_obs()
        self.in_channels = initial_obs.shape[0]  # 6 channels
        self.grid_size = initial_obs.shape[1]  # 5x5 grid
        self.act_dim = 5  # 0: nothing, 1: up, 2: right, 3: down, 4: left

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters (Kept identical to your teammate's setup)
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.99
        self.learning_rate = 5e-5
        self.epsilon_start = 1.0
        self.epsilon_end = 0.10
        self.epsilon_decay_steps = 6000
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # Replay buffer parameters
        self.replay_buffer_capacity = 80000
        self.min_replay_buffer_size = 2000
        self.replay_buffer = []

        # Build networks
        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_q_network(self):
        network = DuelingQNetwork_CNN(self.in_channels, self.act_dim)
        return network

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        else:
            # FIX 1: Convert the 3D NumPy array cleanly to a tensor and add batch dimension [1, 4, 5, 5]
            obs_tensor = torch.from_numpy(obs).float().clone().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            act = torch.argmax(q_values).item()
            return int(act)

    def optimize_model(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.from_numpy(np.stack(next_obs_batch)).float().clone().to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Double DQN target computation safely wrapped in no_grad
        with torch.no_grad():
            # 1. Online network selects actions
            next_q_values_online = self.q_network(next_obs_batch)
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

            # 2. Target network evaluates those specific actions
            next_q_values_target = self.target_network(next_obs_batch)
            max_target_q_values_next = next_q_values_target.gather(1, next_actions).squeeze(1)

            # 3. Calculate bellman equation target
            target_q_values = rew_batch + (1 - done_batch) * self.gamma * max_target_q_values_next

        # Current Q-Values extraction
        q_values = self.q_network(obs_batch)
        action_masks = torch.nn.functional.one_hot(act_batch, num_classes=self.act_dim)
        q_values_for_actions = (q_values * action_masks).sum(dim=1)

        # Compute Loss (target_q_values is already completely detached now)
        loss = self.loss_fn(q_values_for_actions, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def store_transition(self, obs, act, rew, next_obs, done):
        if len(self.replay_buffer) >= self.replay_buffer_capacity:
            self.replay_buffer.pop(0)
        # Raw NumPy arrays from your get_cnn_obs() flow into the list safely
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


# DQN network with pytorch baseline #revised hyperparameters
class DQN_v6:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_v6.'
        self.network_name = 'DQN Network (Version 6)'

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
        self.gamma = 0.99
        self.learning_rate = 5e-5
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = self.num_episodes / 2
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # replay buffer parameters
        self.replay_buffer_capacity = 80000
        self.min_replay_buffer_size = 2000
        self.replay_buffer = []  # initialize replay buffer as empty list

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_q_network(self):
        network = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        with torch.no_grad():
            target_q_values_next = self.target_network(next_obs_batch)
            max_target_q_values_next = target_q_values_next.max(dim=1)[0]
            target_q_values = rew_batch + (1 - done_batch) * self.gamma * max_target_q_values_next

        q_values = self.q_network(obs_batch)
        # the action taken
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

class DQN_CNN_v4:
    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_CNN_v4.'
        self.network_name = 'DQN Convolutional Network (Version 4)'

        # 1. Get initial observation shape dynamically
        initial_obs = self.env.reset('training')  # Returns shape (6, 5, 5) from get_cnn_obs()
        self.in_channels = initial_obs.shape[0]  # 6 channels
        self.grid_size = initial_obs.shape[1]  # 5x5 grid
        self.act_dim = 5  # 0: nothing, 1: up, 2: right, 3: down, 4: left

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters (Kept identical to your teammate's setup)
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.99
        self.learning_rate = 5e-5
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 7000
        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # Replay buffer parameters
        self.replay_buffer_capacity = 80000
        self.min_replay_buffer_size = 8000
        self.replay_buffer = []

        # Build networks
        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_q_network(self):
        # Sweeping 5x5 map space using customized 16 -> 32 filters layout
        network = nn.Sequential(
            # Layer 1: Input (6, 5, 5) -> Output (16, 5, 5)
            nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: Input (16, 5, 5) -> Output (32, 3, 3)  [Valid padding shrinks spatial dimensions to 3x3]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Flatten Layer: Converts (32, 3, 3) feature map to a flat 288 vector
            nn.Flatten(),
            # Layer 3: Decision Dense Layer
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            # Layer 4: Output Layer maps to action dimensions
            nn.Linear(128, self.act_dim)
        )
        return network

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        else:
            # FIX 1: Convert the 3D NumPy array cleanly to a tensor and add batch dimension [1, 6, 5, 5]
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            act = torch.argmax(q_values).item()
            return int(act)

    def optimize_model(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        # FIX 2: Use np.stack() to pack a list of independent 3D arrays into solid blocks of shapes [128, 6, 5, 5]
        obs_batch = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.from_numpy(np.stack(next_obs_batch)).float().to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        with torch.no_grad():
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
        # Raw NumPy arrays from your get_cnn_obs() flow into the list safely
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

class DQN_CNN_v5:
    def __init__(self, env, pretrained_path=None, first_training=True):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f'DQN_CNN_v5.'
        self.network_name = 'DQN Convolutional Network (Version 5)'

        # 1. Get initial observation shape dynamically
        initial_obs = self.env.reset('training')  # Returns shape (6, 5, 5) from get_cnn_obs()
        self.in_channels = initial_obs.shape[0]  # 7 channels
        self.grid_size = initial_obs.shape[1]  # 5x5 grid
        self.act_dim = 5  # 0: nothing, 1: up, 2: right, 3: down, 4: left

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters (Kept identical to your teammate's setup)
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 5e-5
        if first_training:
            self.epsilon_start = 1.0
            self.epsilon_end = 0.05
            self.epsilon_decay_steps = 4000
        else:
            self.epsilon_start = 0.5
            self.epsilon_decay_steps = 6000
            self.epsilon_end = 0.05

        self.epsilon = self.epsilon_start
        self.target_update_freq = 10

        # Replay buffer parameters
        self.replay_buffer_capacity = 20000
        self.min_replay_buffer_size = 2000
        self.replay_buffer = []

        # Build networks
        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        if pretrained_path is not None:
            self.q_network.load_state_dict(
                torch.load(pretrained_path, map_location=self.device))
            print(f'Loaded pretrained weights from {pretrained_path}')

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_q_network(self):
        # Sweeping 5x5 map space using customized 16 -> 32 filters layout
        network = nn.Sequential(
            # Layer 1: Input (7, 5, 5) -> Output (16, 5, 5)
            nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: Input (16, 5, 5) -> Output (32, 3, 3)  [Valid padding shrinks spatial dimensions to 3x3]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Flatten Layer: Converts (32, 3, 3) feature map to a flat 288 vector
            nn.Flatten(),
            # Layer 3: Decision Dense Layer
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            # Layer 4: Output Layer maps to action dimensions
            nn.Linear(128, self.act_dim)
        )
        return network

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        else:
            # FIX 1: Convert the 3D NumPy array cleanly to a tensor and add batch dimension [1, 6, 5, 5]
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            act = torch.argmax(q_values).item()
            return int(act)

    def optimize_model(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        # FIX 2: Use np.stack() to pack a list of independent 3D arrays into solid blocks of shapes [128, 6, 5, 5]
        obs_batch = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        next_obs_batch = torch.from_numpy(np.stack(next_obs_batch)).float().to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        with torch.no_grad():
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
        # Raw NumPy arrays from your get_cnn_obs() flow into the list safely
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
