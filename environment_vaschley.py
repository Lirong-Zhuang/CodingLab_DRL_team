import numpy as np

from environment import Environment


class Environment_vaschley(Environment):
    """Ashley CNN-v2 variant-2 environment with the original 7-channel observation."""

    def __init__(self, variant, data_dir):
        if variant != 2:
            raise ValueError("Environment_vaschley is only intended for variant 2.")
        super().__init__(variant, data_dir)
        self.env_name = "aschley."

    def get_obs(self):
        obs = np.zeros(
            (7, self.vertical_cell_count, self.horizontal_cell_count),
            dtype=np.float32,
        )

        # Channels must stay identical to dev_ashley_new.get_cnn_obs_variant_2.
        obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0
        obs[1, self.target_loc[0], self.target_loc[1]] = 1.0

        if self.item_locs:
            locs = np.asarray(self.item_locs, dtype=np.int32)
            times = np.asarray(self.item_times, dtype=np.float32)
            rows, cols = locs[:, 0], locs[:, 1]
            obs[2, rows, cols] = 1.0
            obs[3, rows, cols] = (self.max_response_time - times) / self.max_response_time

        if self.agent_load > 0:
            obs[4, :, :] = 1.0

        obs[5, :, :] = (self.episode_steps - self.step_count) / self.episode_steps

        eligible_cells = set(self.eligible_cells)
        for row in range(self.vertical_cell_count):
            for col in range(self.horizontal_cell_count):
                if (row, col) not in eligible_cells:
                    obs[6, row, col] = 1.0

        return obs
