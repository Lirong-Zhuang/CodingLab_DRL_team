# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

import random
import os
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np


class RainbowCNNEnvironment(object):
    def __init__(self, variant, data_dir, test_data_dir=None):
        self.variant = variant
        self.vertical_cell_count = 5
        self.horizontal_cell_count = 5
        self.vertical_idx_target = 2
        self.horizontal_idx_target = 0
        self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.episode_steps = 200
        self.max_response_time = 15 if self.variant == 2 else 10
        self.reward = 25 if self.variant == 2 else 15
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.env_name = "rainbow_cnn."

        self.training_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/training_episodes.csv')
        self.training_episodes = self.training_episodes.training_episodes.tolist()
        self.validation_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/validation_episodes.csv')
        self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
        self.test_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/test_episodes.csv')
        self.test_episodes = self.test_episodes.test_episodes.tolist()
        if self.test_data_dir is not None:
            final_test_dir = os.path.join(self.test_data_dir, f'variant_{self.variant}')
            self.test_episodes = sorted(
                int(file_name[len('episode_'):-len('.csv')])
                for file_name in os.listdir(final_test_dir)
                if file_name.startswith('episode_') and file_name.endswith('.csv')
            )

        self.remaining_training_episodes = deepcopy(self.training_episodes)
        self.validation_episode_counter = 0

        if self.variant == 0 or self.variant == 2:
            self.agent_capacity = 1
        else:
            self.agent_capacity = 3

        if self.variant == 0 or self.variant == 1:
            self.eligible_cells = [(0,0), (0,1), (0,2), (0,3), (0,4),
                                   (1,0), (1,1), (1,2), (1,3), (1,4),
                                   (2,0), (2,1), (2,2), (2,3), (2,4),
                                   (3,0), (3,1), (3,2), (3,3), (3,4),
                                   (4,0), (4,1), (4,2), (4,3), (4,4)]
        else:
            self.eligible_cells = [(0,0),        (0,2), (0,3), (0,4),
                                   (1,0),        (1,2),        (1,4),
                                   (2,0),        (2,2),        (2,4),
                                   (3,0), (3,1), (3,2),        (3,4),
                                   (4,0), (4,1), (4,2),        (4,4)]

    def get_dist_from_coord(self, coord1, coord2):
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    # initialize a new episode (specify if training, validation, or testing via the mode argument)
    def reset(self, mode):
        modes = ['training', 'validation', 'testing']
        if mode not in modes:
            raise ValueError('Invalid mode. Expected one of: %s' % modes)

        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.item_locs = []
        self.item_times = []

        if mode == "testing":
            episode = self.test_episodes[0]
            self.test_episodes.remove(episode)
        elif mode == "validation":
            episode = self.validation_episodes[self.validation_episode_counter]
            self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
        else:
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(episode)
        if mode == "testing" and self.test_data_dir is not None:
            episode_path = os.path.join(
                self.test_data_dir,
                f'variant_{self.variant}',
                f'episode_{episode:03d}.csv',
            )
        else:
            episode_path = os.path.join(
                self.data_dir,
                f'variant_{self.variant}',
                'episode_data',
                f'episode_{episode:03d}.csv',
            )
        self.data = pd.read_csv(episode_path, index_col=0)

        return self.get_obs()

    # take one environment step based on the action act
    def step(self, act):
        self.step_count += 1

        rew = 0

        # done signal (1 if episode ends, 0 if not)
        if self.step_count == self.episode_steps:
            done = 1
        else:
            done = 0

        # agent movement
        if act != 0:
            if act == 1:  # up
                new_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            elif act == 2:  # right
                new_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
            elif act == 3:  # down
                new_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            elif act == 4:  # left
                new_loc = (self.agent_loc[0], self.agent_loc[1] - 1)

            if new_loc in self.eligible_cells:
                self.agent_loc = new_loc
                rew += -1

        # item pick-up
        if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
                self.agent_load += 1
                idx = self.item_locs.index(self.agent_loc)
                self.item_locs.pop(idx)
                self.item_times.pop(idx)
                rew += self.reward / 2

        # item drop-off
        if self.agent_loc == self.target_loc:
            rew += self.agent_load * self.reward / 2
            self.agent_load = 0

        # track how long ago items appeared
        self.item_times = [i + 1 for i in self.item_times]

        # remove items for which max response time is reached
        mask = [i < self.max_response_time for i in self.item_times]
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # add items which appear in the current time step
        new_items = self.data[self.data.step == self.step_count]
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        # get new observation
        next_obs = self.get_obs()

        return rew, next_obs, done

    # Seven-channel CNN observation used by the release Rainbow agents.
    def get_obs(self):
        obs = np.zeros(
            (7, self.vertical_cell_count, self.horizontal_cell_count),
            dtype=np.float32,
        )
        obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0
        obs[1, :, :] = self.agent_load / self.agent_capacity
        obs[2, :, :] = (self.episode_steps - self.step_count) / self.episode_steps

        for item_loc, item_time in zip(self.item_locs, self.item_times):
            row, column = item_loc
            time_left = (self.max_response_time - item_time) / self.max_response_time
            distance_to_item = self.get_dist_from_coord(self.agent_loc, item_loc)
            distance_to_target = self.get_dist_from_coord(item_loc, self.target_loc)
            reachability = (self.max_response_time - item_time - distance_to_item) / self.max_response_time
            heuristic_value = (self.reward - distance_to_item - distance_to_target) / self.reward

            obs[3, row, column] = 1.0
            obs[4, row, column] = np.clip(time_left, 0.0, 1.0)
            obs[5, row, column] = np.clip(reachability, 0.0, 1.0)
            obs[6, row, column] = np.clip(heuristic_value, 0.0, 1.0)
        return obs