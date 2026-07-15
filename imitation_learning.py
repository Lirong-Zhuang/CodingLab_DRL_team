import os
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class PretrainVariant2:
    """
    Imitation-learning warm start for the Variant 2 DQN.

    Trains ONLY with TD updates (no margin loss, no epsilon, no action
    selection) on a fixed set of optimal demonstration transitions, so that
    eligible-cell Q-values are pulled positive while never-demonstrated
    (s,a) pairs -- including wall collisions -- stay near their zero-init
    starting point. See conversation history for the full rationale.
    """

    def __init__(self, experiences, zero_init_output=True,
                 batch_size=64, gamma=0.95, learning_rate=5e-5,
                 target_update_freq_steps=500):
        self.file_name = 'Pretrain_Variant_2.'
        self.network_name = 'Imitation Learning - Pretrained DQN Convolutional Network'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        # pretrain has no episodes, so target sync is step-based instead of
        # episode-based; tune this relative to how many batches/epoch you have
        self.target_update_freq_steps = target_update_freq_steps

        self.act_dim = 5
        self.experiences = experiences
        self.in_channels = experiences[0][0].shape[0]
        self.grid_size = experiences[0][0].shape[1]

        self.pretrain_loader = DataLoader(
            self.experiences, batch_size=self.batch_size,
            shuffle=True, collate_fn=self._collate
        )

        self.q_network = self.build_q_network().to(self.device)
        self.target_network = self.build_q_network().to(self.device)

        if zero_init_output:
            # last layer only -- hidden layers keep normal random init so
            # symmetry-breaking still works; only the Q-value readout starts
            # at exactly 0 for every (s,a)
            last_layer = self.q_network[-1]
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def _collate(batch):
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)
        return (
            torch.from_numpy(np.stack(obs_batch)).float(),
            torch.as_tensor(act_batch, dtype=torch.long),
            torch.as_tensor(rew_batch, dtype=torch.float32),
            torch.from_numpy(np.stack(next_obs_batch)).float(),
            torch.as_tensor(done_batch, dtype=torch.float32),
        )

    def build_q_network(self):
        network = nn.Sequential(
            # Layer 1: Input (in_channels, 5, 5) -> Output (16, 5, 5)
            nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: Input (16, 5, 5) -> Output (32, 3, 3)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, self.act_dim)
        )
        return network

    def optimize_step(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch):
        obs_batch = obs_batch.to(self.device)
        act_batch = act_batch.to(self.device)
        rew_batch = rew_batch.to(self.device)
        next_obs_batch = next_obs_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        with torch.no_grad():
            target_q_values_next = self.target_network(next_obs_batch)
            max_target_q_values_next = target_q_values_next.max(dim=1)[0]
            target_q_values = rew_batch + (1 - done_batch) * self.gamma * max_target_q_values_next

        q_values = self.q_network(obs_batch)
        q_values_for_actions = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_values_for_actions, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def pretrain(self, num_epochs, log_dir='./logs/pretrain_variant_2', output_interval=10):
        os.makedirs(log_dir, exist_ok=True) if not os.path.exists(log_dir) else None
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text(
            'Hyperparameters',
            f'batch_size: {self.batch_size}, gamma: {self.gamma}, '
            f'learning_rate: {self.learning_rate}, num_epochs: {num_epochs}, '
            f'num_transitions: {len(self.experiences)}',
            0
        )

        start_time = time.time()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_losses = []
            for obs_b, act_b, rew_b, next_obs_b, done_b in self.pretrain_loader:
                loss = self.optimize_step(obs_b, act_b, rew_b, next_obs_b, done_b)
                epoch_losses.append(loss)
                global_step += 1

                if global_step % self.target_update_freq_steps == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            writer.add_scalar('Loss/pretrain', avg_loss, epoch + 1)


            elapsed = time.time() - start_time
            print(f'Epoch {epoch + 1}/{num_epochs} | avg loss: {avg_loss:.4f} | elapsed: {elapsed:.1f}s')

        # final sync so target_network reflects the fully pretrained weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        writer.close()
        print(f'Pretraining complete. Total time: {time.time() - start_time:.1f}s')

    def save(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.q_network.state_dict(), model_path)
        print(f'Saved pretrained model to {model_path}')


if __name__ == '__main__':
    with open('./demonstrations/demo_v2.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f'Loaded {len(data)} demo transitions')
    print(f'obs shape: {data[0][0].shape}')

    pretrainer = PretrainVariant2(data, zero_init_output=False)
    pretrainer.pretrain(num_epochs=50)
    pretrainer.save('./models/pretrained_variant_2.pt')
