import random
from copy import deepcopy

import pandas as pd

from environment_v9 import Environment_v9


class Environment_v91(Environment_v9):
    """Environment v9 using the focused variant-0 validation episode list."""

    def __init__(self, variant, data_dir):
        super().__init__(variant, data_dir)
        self.env_name = "91."
        self.validation_episodes = pd.read_csv(
            self.data_dir + f'/variant_{self.variant}/validation_episodes_focus1.csv'
        )
        self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
        self.validation_episode_counter = 0

    def reset(self, mode):
        modes = ['training', 'validation', 'testing']
        if mode not in modes:
            raise ValueError('Invalid mode. Expected one of: %s' % modes)

        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0
        self.item_locs = []
        self.item_times = []

        if mode == "testing":
            episode = self.test_episodes[0]
            self.test_episodes.remove(episode)
        elif mode == "validation":
            episode = self.validation_episodes[self.validation_episode_counter]
            self.validation_episode_counter = (
                self.validation_episode_counter + 1
            ) % len(self.validation_episodes)
        else:
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(episode)

        self.data = pd.read_csv(
            self.data_dir + f'/variant_{self.variant}/episode_data/episode_{episode:03d}.csv',
            index_col=0
        )

        return self.get_obs()
