import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rainbow_dqn import NoisyLinear, PrioritizedReplayBuffer


class UniformReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=None):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        samples = [self.buffer[i] for i in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        return None

    def __len__(self):
        return len(self.buffer)


class AblationCNNQNetwork(nn.Module):
    def __init__(self, in_channels, act_dim, num_atoms=51, use_c51=True, use_noisy=True, use_dueling=True):
        super().__init__()
        self.act_dim = act_dim
        self.num_atoms = num_atoms
        self.use_c51 = use_c51
        self.use_noisy = use_noisy
        self.use_dueling = use_dueling
        output_dim = num_atoms if use_c51 else 1
        linear = NoisyLinear if use_noisy else nn.Linear

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.feature = nn.Sequential(
            nn.Linear(32 * 3 * 3, 32),
            nn.ReLU(),
        )

        if use_dueling:
            self.value_stream = nn.Sequential(
                linear(32, 32),
                nn.ReLU(),
                linear(32, output_dim),
            )
            self.advantage_stream = nn.Sequential(
                linear(32, 32),
                nn.ReLU(),
                linear(32, act_dim * output_dim),
            )
        else:
            self.q_stream = nn.Sequential(
                linear(32, 32),
                nn.ReLU(),
                linear(32, act_dim * output_dim),
            )

    def forward(self, x):
        features = self.feature(self.cnn(x))

        if self.use_dueling:
            value = self.value_stream(features).view(-1, 1, self.num_atoms if self.use_c51 else 1)
            advantage = self.advantage_stream(features).view(
                -1,
                self.act_dim,
                self.num_atoms if self.use_c51 else 1,
            )
            output = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            output = self.q_stream(features).view(-1, self.act_dim, self.num_atoms if self.use_c51 else 1)

        if self.use_c51:
            return torch.softmax(output, dim=-1)

        return output.squeeze(-1)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RainbowDQNAblation:
    ablation_id = "a"
    ablation_name = "Rainbow DQN Ablation"
    use_prioritized_replay = True
    use_multi_step_return = True
    use_c51 = True
    use_noisy = True
    use_double = True
    use_dueling = True

    def __init__(self, env):
        self.env = env
        self.variant = self.env.variant
        self.data_dir = self.env.data_dir
        self.file_name = f"DQN_{self.ablation_id}."
        self.network_name = self.ablation_name
        self.training_step = 0

        initial_obs = self.env.reset("training")
        self.in_channels = initial_obs.shape[0]
        self.grid_height = initial_obs.shape[1]
        self.grid_width = initial_obs.shape[2]
        self.act_dim = 5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

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

        self.epsilon_start = 0.0 if self.use_noisy else 1.0
        self.epsilon_end = 0.0 if self.use_noisy else 0.05
        self.epsilon_decay_steps = 8000
        self.epsilon = self.epsilon_start

        self.n_step = 3 if self.use_multi_step_return else 1
        self.n_step_buffer = []

        self.replay_buffer_capacity = 50000
        self.min_replay_buffer_size = 1000
        self.per_alpha = 0.6
        self.per_beta_start = 0.4
        self.per_beta_end = 1.0
        self.per_beta = self.per_beta_start
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, self.per_alpha)
        else:
            self.replay_buffer = UniformReplayBuffer(self.replay_buffer_capacity)

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def build_q_network(self):
        return AblationCNNQNetwork(
            self.in_channels,
            self.act_dim,
            self.num_atoms,
            use_c51=self.use_c51,
            use_noisy=self.use_noisy,
            use_dueling=self.use_dueling,
        )

    def _q_values(self, network_output):
        if self.use_c51:
            return (network_output * self.support.view(1, 1, -1)).sum(dim=2)
        return network_output

    def select_action(self, obs):
        if not self.use_noisy and np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self._q_values(self.q_network(obs_tensor))
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

        if self.use_c51:
            losses = self._distributional_losses(obs_batch, act_batch, rew_batch, next_obs_batch, done_batch)
        else:
            losses = self._td_losses(obs_batch, act_batch, rew_batch, next_obs_batch, done_batch)

        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        if self.use_prioritized_replay:
            new_priorities = losses.detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, new_priorities)

        self.q_network.reset_noise()
        self.target_network.reset_noise()
        self.training_step += 1

        return loss.item()

    def _distributional_losses(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch):
        batch_indices = torch.arange(self.batch_size).to(self.device)

        with torch.no_grad():
            if self.use_double:
                next_dist_online = self.q_network(next_obs_batch)
                next_q_online = self._q_values(next_dist_online)
                next_actions = next_q_online.argmax(dim=1)
            else:
                next_dist_target_all = self.target_network(next_obs_batch)
                next_q_target = self._q_values(next_dist_target_all)
                next_actions = next_q_target.argmax(dim=1)

            next_dist_target = self.target_network(next_obs_batch)
            next_dist = next_dist_target[batch_indices, next_actions]

            target_support = rew_batch.unsqueeze(1) + (
                1 - done_batch.unsqueeze(1)
            ) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)
            target_support = target_support.clamp(self.v_min, self.v_max)

            b = (target_support - self.v_min) / self.delta_z
            lower = b.floor().long().clamp(0, self.num_atoms - 1)
            upper = b.ceil().long().clamp(0, self.num_atoms - 1)
            projected_dist = torch.zeros_like(next_dist)

            lower_weight = upper.float() - b
            upper_weight = b - lower.float()
            exact_match = upper == lower
            lower_weight[exact_match] = 1.0
            upper_weight[exact_match] = 0.0

            offset = (torch.arange(self.batch_size) * self.num_atoms).unsqueeze(1).to(self.device)
            projected_dist.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * lower_weight).view(-1))
            projected_dist.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * upper_weight).view(-1))

        dist = self.q_network(obs_batch)
        chosen_dist = dist[batch_indices, act_batch].clamp(min=1e-8)
        return -(projected_dist * torch.log(chosen_dist)).sum(dim=1)

    def _td_losses(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch):
        with torch.no_grad():
            if self.use_double:
                next_actions = self.q_network(next_obs_batch).argmax(dim=1)
                next_q_values = self.target_network(next_obs_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_obs_batch).max(dim=1).values
            target_q_values = rew_batch + (1 - done_batch) * (self.gamma ** self.n_step) * next_q_values

        current_q_values = self.q_network(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze(1)
        return self.loss_fn(current_q_values, target_q_values)

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

        if not self.use_noisy:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps,
            )

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)


class DQN_a0(RainbowDQNAblation):
    ablation_id = "a0"
    ablation_name = "Rainbow DQN Ablation a0: full baseline"


class DQN_a1(RainbowDQNAblation):
    ablation_id = "a1"
    ablation_name = "Rainbow DQN Ablation a1: w/o prioritized replay"
    use_prioritized_replay = False


class DQN_a2(RainbowDQNAblation):
    ablation_id = "a2"
    ablation_name = "Rainbow DQN Ablation a2: w/o multi-step return"
    use_multi_step_return = False


class DQN_a3(RainbowDQNAblation):
    ablation_id = "a3"
    ablation_name = "Rainbow DQN Ablation a3: w/o C51"
    use_c51 = False


class DQN_a4(RainbowDQNAblation):
    ablation_id = "a4"
    ablation_name = "Rainbow DQN Ablation a4: w/o noisy network"
    use_noisy = False


class DQN_a5(RainbowDQNAblation):
    ablation_id = "a5"
    ablation_name = "Rainbow DQN Ablation a5: w/o double DQN (dueling only)"
    use_double = False
