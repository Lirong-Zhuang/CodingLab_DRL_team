import numpy as np

from .environment import Environment


class Environment(Environment):
    def __init__(self, variant, data_dir):
        super().__init__(variant, data_dir)

    def get_dist_from_coord(self, coord1, coord2):
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    # CNN observation shape: 7 x 5 x 5
    def get_obs(self):
        obs = np.zeros(
            (7, self.vertical_cell_count, self.horizontal_cell_count),
            dtype=np.float32,
        )

        # Channel 0: agent location
        obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: agent load, broadcast to the whole grid
        obs[1, :, :] = self.agent_load / self.agent_capacity

        # Channel 2: remaining episode steps, broadcast to the whole grid
        remaining_steps = (self.episode_steps - self.step_count) / self.episode_steps
        obs[2, :, :] = remaining_steps

        # Channel 3: item presence
        # Channel 4: item remaining time
        # Channel 5: item reachability before its response time expires
        # Channel 6: normalized heuristic value:
        #            (reward - agent-item distance - item-target distance) / reward
        for item_loc, item_time in zip(self.item_locs, self.item_times):
            r, c = item_loc
            time_left = (
                self.max_response_time - item_time
            ) / self.max_response_time
            time_left = max(0.0, min(1.0, time_left))

            dist_agent_item = self.get_dist_from_coord(self.agent_loc, item_loc)
            dist_item_target = self.get_dist_from_coord(item_loc, self.target_loc)

            remaining_item_steps = self.max_response_time - item_time
            reachability = (
                remaining_item_steps - dist_agent_item
            ) / self.max_response_time
            reachability = max(0.0, min(1.0, reachability))

            heuristic_value = (
                self.reward - dist_agent_item - dist_item_target
            ) / self.reward
            heuristic_value = max(0.0, min(1.0, heuristic_value))

            obs[3, r, c] = 1.0
            obs[4, r, c] = time_left
            obs[5, r, c] = reachability
            obs[6, r, c] = heuristic_value

        return obs
