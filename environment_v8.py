import numpy as np

from environment_v6 import Environment_v6


class Environment_v8(Environment_v6):
    # Same observation layout as Environment_v6, but time features use elapsed
    # values instead of remaining values.
    def __init__(self, variant, data_dir):
        super().__init__(variant, data_dir)
        self.env_name = "8."

    def get_obs(self):
        # CNN input: 3 x 5 x 5
        grid_obs = np.zeros(
            (3, self.vertical_cell_count, self.horizontal_cell_count),
            dtype=np.float32
        )

        # Channel 0: agent location
        grid_obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: item presence
        # Channel 2: item elapsed time / urgency
        for item_loc, item_time in zip(self.item_locs, self.item_times):
            r, c = item_loc
            urgency = item_time / self.max_response_time
            urgency = max(0.0, min(1.0, urgency))

            grid_obs[1, r, c] = 1.0
            grid_obs[2, r, c] = urgency

        # MLP input: 2 scalar features
        scalar_obs = np.array([
            self.agent_load / self.agent_capacity,
            self.step_count / self.episode_steps
        ], dtype=np.float32)

        return grid_obs, scalar_obs
