import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_logits = self.actor(state)
        state_value = self.critic(state)
        return action_logits, state_value


# Improved PPO agent
class PPO_v1:
    def __init__(self, env):
        self.env = env
        self.variant = env.variant
        self.data_dir = env.data_dir
        self.file_name = "PPO_v1."
        self.network_name = "PPO Network (Version 1)"

        initial_obs = self.env.reset("training")
        self.obs_dim = len(initial_obs)
        self.act_dim = 5

        # Training device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Training settings
        self.episode_steps = self.env.episode_steps
        self.num_episodes = 10000

        self.gamma = 0.95
        self.learning_rate = 1e-4
        self.clip_epsilon = 0.2
        self.update_epochs = 4
        self.rollout_episodes = 20
        self.rollout_steps = self.episode_steps * self.rollout_episodes
        self.entropy_coef = 0.05
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        # Network and optimizer
        self.network = ActorCritic(
            self.obs_dim,
            self.act_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, state_value = self.network(obs_tensor)

        action_dist = torch.distributions.Categorical(
            logits=action_logits
        )

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item(), state_value.item()

    def compute_returns(self, rewards, dones, values):
        returns = []
        R = 0.0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0.0

            R = reward + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        advantages = returns - values

        if len(advantages) > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self, obs, actions, log_probs, returns, advantages):
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)

        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.update_epochs):
            action_logits, state_values = self.network(obs_tensor)

            action_dist = torch.distributions.Categorical(
                logits=action_logits
            )

            new_log_probs = action_dist.log_prob(actions_tensor)
            entropy = action_dist.entropy().mean()

            ratio = torch.exp(
                new_log_probs - old_log_probs_tensor
            )

            unclipped_objective = ratio * advantages

            clipped_objective = torch.clamp(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon
            ) * advantages

            actor_loss = -torch.min(
                unclipped_objective,
                clipped_objective
            ).mean()

            critic_loss = nn.MSELoss()(
                state_values.squeeze(),
                returns
            )

            loss = (
                actor_loss
                + self.value_coef * critic_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.max_grad_norm
            )

            self.optimizer.step()

            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()

        n = self.update_epochs

        return (
            total_loss / n,
            total_actor_loss / n,
            total_critic_loss / n,
            total_entropy / n
        )

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "variant": self.variant,
                "agent": self.network_name,
                "gamma": self.gamma,
                "learning_rate": self.learning_rate,
                "clip_epsilon": self.clip_epsilon,
                "update_epochs": self.update_epochs,
            },
            path
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.network.load_state_dict(checkpoint)

        self.network.eval()