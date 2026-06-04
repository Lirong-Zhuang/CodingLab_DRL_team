import numpy as np

from environment_v5 import Environment_v5


class Environment_v7(Environment_v5):
    # Same observation layout as Environment_v5, but time features use elapsed
    # values instead of remaining values.
    def __init__(self, variant, data_dir):
        super().__init__(variant, data_dir)
        self.env_name = "7."

    def get_obs(self):
        # CNN input: 5 x 5 x 5
        obs = np.zeros(
            (5, self.vertical_cell_count, self.horizontal_cell_count),
            dtype=np.float32
        )

        # Channel 0: agent location
        obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: agent load, broadcast to whole grid
        obs[1, :, :] = self.agent_load / self.agent_capacity

        # Channel 2: elapsed episode steps, broadcast to whole grid
        elapsed_steps = self.step_count / self.episode_steps
        obs[2, :, :] = elapsed_steps

        # Channel 3: item presence
        # Channel 4: item elapsed time / urgency
        for item_loc, item_time in zip(self.item_locs, self.item_times):
            r, c = item_loc
            urgency = item_time / self.max_response_time
            urgency = max(0.0, min(1.0, urgency))

            obs[3, r, c] = 1.0
            obs[4, r, c] = urgency

        return obs
