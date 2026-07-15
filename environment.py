# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

import random
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np
from spatial_distribution import compute_spatial_distribution


class Environment(object):
    def __init__(self, variant, data_dir, data_variant=None):
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
        if data_variant is None:
            self.data_variant = variant
        else:
            self.data_variant = data_variant

        self.training_episodes = pd.read_csv(self.data_dir + f'/variant_{self.data_variant}/training_episodes.csv')
        self.training_episodes = self.training_episodes.training_episodes.tolist()
        self.validation_episodes = pd.read_csv(self.data_dir + f'/variant_{self.data_variant}/validation_episodes.csv')
        self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
        self.test_episodes = pd.read_csv(self.data_dir + f'/variant_{self.data_variant}/test_episodes.csv')
        self.test_episodes = self.test_episodes.test_episodes.tolist()

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

        _, normalized_prob, _ = compute_spatial_distribution(self.variant, './data')
        self.spawn_channel = np.zeros(
            (self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)
        for (r, c), prob in normalized_prob.items():
            self.spawn_channel[r, c] = prob



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
        self.data = pd.read_csv(self.data_dir + f'/variant_{self.data_variant}/episode_data/episode_{episode:03d}.csv',
                                index_col=0)

        return self.get_cnn_obs_distribution_variant_2()

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
        # checks if each item's age is strictly less than max_response_time (which is 10 for Variant 0) and creates a list of True and False values.
        mask = [i < self.max_response_time for i in self.item_times]
        # compress() function: itertools.compress takes a data list and a boolean mask list.
        # It keeps the data elements where the mask is True, and throws away the ones where it is False
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # add items which appear in the current time step
        new_items = self.data[self.data.step == self.step_count]
        # uses zip() to combine the vertical and horizontal columns into neat tuple pairs like (row, col)
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
        # if new_items contains multiple new coordinates, they are all thrown into self.item_locs at the same time
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        # get new observation
        next_obs = self.get_cnn_obs_distribution_variant_2()

        return rew, next_obs, done

    # TODO: implement function that gives the input features for the neural network(s)
    #       based on the current state of the environment
    # Using one-hot and multi-hot to realize the fixed input size. Dimension of obs is 54. Numbers are normalized
    def get_obs(self):
        
        # agent_loc_obs = [0] * self.vertical_cell_count * self.horizontal_cell_count # 25-dimensional one-hot agent location as input
        # agent_loc_obs[self.agent_loc[0] * self.horizontal_cell_count + self.agent_loc[1]] = 1
        agent_loc_obs = [self.agent_loc[0] / self.vertical_cell_count, self.agent_loc[1] / self.horizontal_cell_count]  # 2-dimensional agent location as input 

        item_loc_obs = [0] * self.vertical_cell_count * self.horizontal_cell_count
        item_times_obs = [0] * self.vertical_cell_count * self.horizontal_cell_count

        for item_loc, item_time in zip(self.item_locs, self.item_times):
            idx = item_loc[0] * self.horizontal_cell_count + item_loc[1]
            item_loc_obs[idx] = 1
            item_times_obs[idx] = item_time / self.max_response_time

        step_count_obs = [self.step_count / self.episode_steps]
        agent_load_obs = [self.agent_load / self.agent_capacity]

        obs = step_count_obs + agent_loc_obs + agent_load_obs + item_loc_obs + item_times_obs

        return obs

    def get_cnn_obs(self):
        # Initialize a 4-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (4, 5, 5)
        grid_obs = np.zeros((4, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2: Available Items (Time-adjusted)
        if self.agent_load == 0:
            for item_loc, item_time in zip(self.item_locs, self.item_times):
                # Professor's logic: item_time starts at 0 when spawned and counts UP to 10.
                # We transform this so: Fresh item = 1.0, about to vanish = 0.1
                steps_remaining = self.max_response_time - item_time
                normalized_time = steps_remaining / self.max_response_time

                # Clip bounds safely between 0.0 and 1.0
                grid_obs[2, item_loc[0], item_loc[1]] = max(0.0, min(1.0, normalized_time))
        else:
            # Agent is already carrying an item!
            # In Variant 0, it cannot pick up anything else, so wipe the item floor map to 0.
            grid_obs[2, :, :] = 0.0

        # Channel 3: Inventory Status
        # Since capacity is 1, this whole layer becomes 1.0 if carrying, 0.0 if empty
        if self.agent_load > 0:
            grid_obs[3, :, :] = 1.0

        return grid_obs

    def get_cnn_obs_v2(self):
        # Initialize a 6-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (6, 5, 5)
        grid_obs = np.zeros((6, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2 & 3: Item Position & Time Left
        if len(self.item_locs) > 0:
            locs_array = np.array(self.item_locs, dtype=np.int32)  # Shape: (N, 2)
            times_array = np.array(self.item_times, dtype=np.float32)  # Shape: (N,)
            vertical_indices = locs_array[:, 0]
            horizontal_indices = locs_array[:, 1]
            steps_remaining = self.max_response_time - times_array
            normalized_time = steps_remaining / self.max_response_time
            clipped_time = np.clip(normalized_time, 0.0, 1.0) #just a safe net
            grid_obs[2, vertical_indices, horizontal_indices] = 1.0
            grid_obs[3, vertical_indices, horizontal_indices] = clipped_time


        # Channel 4: Carrying Status
        # Since capacity is 1, this whole layer becomes 1.0 if carrying, 0.0 if empty
        if self.agent_load > 0:
            grid_obs[4, :, :] = 1.0

        # Channel 5: Step Left (in this episode)
        normalized_step_left = (self.episode_steps - self.step_count) / self.episode_steps
        grid_obs[5, :, :] = normalized_step_left

        return grid_obs

    def get_cnn_obs_variant_1(self):
        # Initialize a 6-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (6, 5, 5)
        grid_obs = np.zeros((6, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2 & 3: Item Position & Time Left
        if len(self.item_locs) > 0:
            locs_array = np.array(self.item_locs, dtype=np.int32)  # Shape: (N, 2)
            times_array = np.array(self.item_times, dtype=np.float32)  # Shape: (N,)
            vertical_indices = locs_array[:, 0]
            horizontal_indices = locs_array[:, 1]
            grid_obs[2, vertical_indices, horizontal_indices] = 1.0
            grid_obs[3, vertical_indices, horizontal_indices] = (self.max_response_time - times_array) / self.max_response_time


        # Channel 4: Loading Capacity Status
        # Similar set-up as time left, when the agent is carrying nothing, the capacity is 1.0,
        # when it carries 1 item, it's 2/3, when it carries 2 item, it's 1/3, when it carries 3, then 0.0
        grid_obs[4, :, :] = (self.agent_capacity - self.agent_load) / self.agent_capacity

        # Channel 5: Step Left (in this episode)
        grid_obs[5, :, :] = (self.episode_steps - self.step_count) / self.episode_steps

        return grid_obs

    def get_cnn_obs_variant_2(self):
        # Initialize a 7-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (7, 5, 5)
        grid_obs = np.zeros((7, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2 & 3: Item Position & Time Left
        if len(self.item_locs) > 0:
            locs_array = np.array(self.item_locs, dtype=np.int32)  # Shape: (N, 2)
            times_array = np.array(self.item_times, dtype=np.float32)  # Shape: (N,)
            vertical_indices = locs_array[:, 0]
            horizontal_indices = locs_array[:, 1]
            grid_obs[2, vertical_indices, horizontal_indices] = 1.0
            grid_obs[3, vertical_indices, horizontal_indices] = (self.max_response_time - times_array) / self.max_response_time

        # Channel 4: Carrying Status
        # Since capacity is 1, this whole layer becomes 1.0 if carrying, 0.0 if empty
        if self.agent_load > 0:
            grid_obs[4, :, :] = 1.0

        # Channel 5: Step Left (in this episode)
        grid_obs[5, :, :] = (self.episode_steps - self.step_count) / self.episode_steps

        # Channel 6: Blocked cells (complement of self.eligible_cells on a 5x5 grid)
        blockage = [(0,1), (1,1), (2,1), (1,3), (2,3), (3,3), (4,3)]
        np_blockage = np.array(blockage, dtype=np.int32)
        blockage_vertical_indices = np_blockage[:, 0]
        blockage_horizontal_indices = np_blockage[:, 1]
        grid_obs[6, blockage_vertical_indices, blockage_horizontal_indices] = 1.0

        return grid_obs

    def get_flattened_obs_variant_2(self):
        # Initialize a 5-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (5, 5, 5)
        grid_obs = np.zeros((5, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2 & 3: Item Position & Time Left
        if len(self.item_locs) > 0:
            locs_array = np.array(self.item_locs, dtype=np.int32)  # Shape: (N, 2)
            times_array = np.array(self.item_times, dtype=np.float32)  # Shape: (N,)
            vertical_indices = locs_array[:, 0]
            horizontal_indices = locs_array[:, 1]
            grid_obs[2, vertical_indices, horizontal_indices] = 1.0
            grid_obs[3, vertical_indices, horizontal_indices] = (self.max_response_time - times_array) / self.max_response_time

        # Channel 4: Blocked cells (complement of self.eligible_cells on a 5x5 grid)
        blockage = [(0,1), (1,1), (2,1), (1,3), (2,3), (3,3), (4,3)]
        np_blockage = np.array(blockage, dtype=np.int32)
        blockage_vertical_indices = np_blockage[:, 0]
        blockage_horizontal_indices = np_blockage[:, 1]
        grid_obs[4, blockage_vertical_indices, blockage_horizontal_indices] = 1.0

        carrying_status = 1.0 if self.agent_load > 0 else 0.0
        normalized_steps_left = (self.episode_steps - self.step_count) / self.episode_steps
        scalar_obs = np.array([carrying_status, normalized_steps_left], dtype=np.float32)

        # 25*5 + carrying_status + normalize_steps_left = 127
        return np.concatenate([grid_obs.flatten(), scalar_obs])

    def get_cnn_obs_realizable_reward_variant_2(self):
        # Initialize a 6-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (6, 5, 5)
        grid_obs = np.zeros((6, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2, 3: Item Position /Realizable Reward
        if len(self.item_locs) > 0:
            locs_array = np.array(self.item_locs, dtype=np.int32)  # Shape: (N, 2)
            times_array = np.array(self.item_times, dtype=np.float32)  # Shape: (N,)
            vertical_indices = locs_array[:, 0]
            horizontal_indices = locs_array[:, 1]

            # Vectorized distance lookups using list comprehension (fast dict access)
            dist_to_item = np.array([
                self.get_shortest_distance(self.agent_loc, (r, c))
                for r, c in self.item_locs
            ], dtype=np.float32)

            dist_item_to_target = np.array([
                self.get_shortest_distance((r, c), self.target_loc)
                for r, c in self.item_locs
            ], dtype=np.float32)

            steps_remaining = self.max_response_time - times_array
            reachable = dist_to_item <= steps_remaining
            if self.step_count > (self.episode_steps - 26):
                steps_left_in_episode = self.episode_steps - self.step_count
                completable = (dist_to_item + dist_item_to_target) <= steps_left_in_episode
                partially_completable = dist_to_item <= steps_left_in_episode
                net_reward_full = self.reward - dist_to_item - dist_item_to_target
                net_reward_partial = (self.reward / 2) - dist_to_item
                net_reward = np.where(
                    reachable & completable,
                    np.maximum(0, net_reward_full),
                    np.where(
                        reachable & partially_completable,
                        np.maximum(0, net_reward_partial),
                        0
                    )
                )
            else:
                net_reward_full = self.reward - dist_to_item - dist_item_to_target
                net_reward = np.where(reachable, np.maximum(0, net_reward_full), 0)
            grid_obs[2, vertical_indices, horizontal_indices] = 1.0
            grid_obs[3, vertical_indices, horizontal_indices] = net_reward / self.reward

        # Channel 4: Carrying Status
        # Since capacity is 1, this whole layer becomes 1.0 if carrying, 0.0 if empty
        if self.agent_load > 0:
            grid_obs[4, :, :] = 1.0

        # Channel 5: eligible cells
        np_eligible = np.array(self.eligible_cells, dtype=np.int32)
        eligible_vertical_indices = np_eligible[:, 0]
        eligible_horizontal_indices = np_eligible[:, 1]
        grid_obs[5, eligible_vertical_indices, eligible_horizontal_indices] = 1.0

        return grid_obs


    def get_shortest_distance(self, start, end):
        """Lazily computes and caches all pairwise shortest distances between eligible cells"""
        if not hasattr(self, '_all_distances'):
            self._all_distances = self._precompute_all_distances()
        return self._all_distances[(start, end)]

    def _precompute_all_distances(self):
        """Computes shortest distance (BFS) between every pair of eligible cells"""
        all_distances = {}
        for start in self.eligible_cells:
            for end in self.eligible_cells:
                all_distances[(start, end)] = self._bfs_distance(start, end)
        return all_distances

    def _bfs_distance(self, start, end):
        """BFS shortest path between two cells, respecting eligible_cells (obstacles)"""
        if start == end:
            return 0

        from collections import deque
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            loc, dist = queue.popleft()
            for neighbor in self._get_neighbors(loc):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf')  # unreachable (shouldn't happen on a connected grid)

    def _get_neighbors(self, loc):
        """Returns eligible adjacent cells (up, down, left, right)"""
        r, c = loc
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [cell for cell in candidates if cell in self.eligible_cells]

    def get_cnn_transfer_obs(self):
        # Initialize a 7-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (7, 5, 5)
        grid_obs = np.zeros((7, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2 & 3: Item Position & Time Left
        if len(self.item_locs) > 0:
            eligible_set = set(self.eligible_cells)
            valid_mask = [loc in eligible_set for loc in self.item_locs]
            valid_item_locs = list(compress(self.item_locs, valid_mask))
            valid_item_times = list(compress(self.item_times, valid_mask))
            locs_array = np.array(valid_item_locs, dtype=np.int32)  # Shape: (N, 2)
            times_array = np.array(valid_item_times, dtype=np.float32)  # Shape: (N,)
            vertical_indices = locs_array[:, 0]
            horizontal_indices = locs_array[:, 1]
            grid_obs[2, vertical_indices, horizontal_indices] = 1.0
            grid_obs[3, vertical_indices, horizontal_indices] = (self.max_response_time - times_array) / self.max_response_time

        # Channel 4: Carrying Status
        # Since capacity is 1, this whole layer becomes 1.0 if carrying, 0.0 if empty
        if self.agent_load > 0:
            grid_obs[4, :, :] = 1.0

        # Channel 5: Step Left (in this episode)
        grid_obs[5, :, :] = (self.episode_steps - self.step_count) / self.episode_steps

        # Channel 6: self.eligible_cells on a 5x5 grid
        np_eligible = np.array(self.eligible_cells, dtype=np.int32)
        eligible_vertical_indices = np_eligible[:, 0]
        eligible_horizontal_indices = np_eligible[:, 1]
        grid_obs[6, eligible_vertical_indices, eligible_horizontal_indices] = 1.0

        return grid_obs

    def get_cnn_obs_distribution_variant_1(self):
        # Initialize a 7-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (7, 5, 5)
        grid_obs = np.zeros((7, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2 & 3: Item Position & Time Left
        if len(self.item_locs) > 0:
            locs_array = np.array(self.item_locs, dtype=np.int32)  # Shape: (N, 2)
            times_array = np.array(self.item_times, dtype=np.float32)  # Shape: (N,)
            vertical_indices = locs_array[:, 0]
            horizontal_indices = locs_array[:, 1]
            grid_obs[2, vertical_indices, horizontal_indices] = 1.0
            grid_obs[3, vertical_indices, horizontal_indices] = (self.max_response_time - times_array) / self.max_response_time


        # Channel 4: Loading Capacity Status
        # Similar set-up as time left, when the agent is carrying nothing, the capacity is 1.0,
        # when it carries 1 item, it's 2/3, when it carries 2 item, it's 1/3, when it carries 3, then 0.0
        grid_obs[4, :, :] = (self.agent_capacity - self.agent_load) / self.agent_capacity

        # Channel 5: Step Left (in this episode)
        grid_obs[5, :, :] = (self.episode_steps - self.step_count) / self.episode_steps

        # Channel 6: Pre-calculated Item Spatial Distribution
        grid_obs[6, :, :] = self.spawn_channel

        return grid_obs

    def get_cnn_obs_constraint_variant_1(self):
        # Initialize a 6-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (6, 5, 5)
        grid_obs = np.zeros((6, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2 & 3: Item Position & Time Left
        if len(self.item_locs) > 0:
            locs_array = np.array(self.item_locs, dtype=np.int32)  # Shape: (N, 2)
            times_array = np.array(self.item_times, dtype=np.float32)  # Shape: (N,)
            vertical_indices = locs_array[:, 0]
            horizontal_indices = locs_array[:, 1]
            grid_obs[2, vertical_indices, horizontal_indices] = 1.0
            steps_till_item_expires = self.max_response_time - times_array
            steps_till_episode_ends = self.episode_steps - self.step_count
            time_constraint = np.minimum(steps_till_item_expires, steps_till_episode_ends)
            grid_obs[3, vertical_indices, horizontal_indices] = time_constraint / self.max_response_time


        # Channel 4: Loading Capacity Status
        # Similar set-up as time left, when the agent is carrying nothing, the capacity is 1.0,
        # when it carries 1 item, it's 2/3, when it carries 2 item, it's 1/3, when it carries 3, then 0.0
        grid_obs[4, :, :] = (self.agent_capacity - self.agent_load) / self.agent_capacity


        # Channel 5: Pre-calculated Item Spatial Distribution
        grid_obs[5, :, :] = self.spawn_channel

        return grid_obs

    def get_cnn_obs_distribution_variant_2(self):
        # Initialize a 8-channel, 5x5 float32 array with zeros
        # Shape: (Channels, Height, Width) -> (8, 5, 5)
        grid_obs = np.zeros((8, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: Agent Position, one-hot spacial map
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: Target Position (Constantly sits at fixed coordinate (2, 0)) , one-hot spacial map
        grid_obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        # Channel 2 & 3: Item Position & Time Left
        if len(self.item_locs) > 0:
            locs_array = np.array(self.item_locs, dtype=np.int32)  # Shape: (N, 2)
            times_array = np.array(self.item_times, dtype=np.float32)  # Shape: (N,)
            vertical_indices = locs_array[:, 0]
            horizontal_indices = locs_array[:, 1]
            grid_obs[2, vertical_indices, horizontal_indices] = 1.0
            steps_till_item_expires = self.max_response_time - times_array
            steps_till_episode_ends = self.episode_steps - self.step_count
            time_constraint = np.minimum(steps_till_item_expires, steps_till_episode_ends)
            grid_obs[3, vertical_indices, horizontal_indices] = time_constraint / self.max_response_time


        # Channel 4: Loading Capacity Status
        # Similar set-up as time left, when the agent is carrying nothing, the capacity is 1.0,
        # when it carries 1 item, it's 2/3, when it carries 2 item, it's 1/3, when it carries 3, then 0.0
        grid_obs[4, :, :] = (self.agent_capacity - self.agent_load) / self.agent_capacity


        # Channel 5: Pre-calculated Item Spatial Distribution
        grid_obs[5, :, :] = self.spawn_channel

        # Channel 6: Global time left
        grid_obs[6, :, :] = (self.episode_steps - self.step_count) / self.episode_steps

        # Channel 7: eligible cell representation
        np_eligible = np.array(self.eligible_cells, dtype=np.int32)
        eligible_vertical_indices = np_eligible[:, 0]
        eligible_horizontal_indices = np_eligible[:, 1]
        grid_obs[7, eligible_vertical_indices, eligible_horizontal_indices] = 1.0

        return grid_obs


