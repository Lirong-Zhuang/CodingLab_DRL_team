import numpy as np

from environment_v5 import Environment_v5


class Environment_v14(Environment_v5):
    """Variant-2 observation without the hand-crafted item-value channel."""

    def __init__(self, variant, data_dir):
        if variant != 2:
            raise ValueError("Environment_v14 is only intended for variant 2.")
        super().__init__(variant, data_dir)
        self.env_name = "14."
        eligible = set(self.eligible_cells)
        self.blocked_cells = [
            (row, col)
            for row in range(self.vertical_cell_count)
            for col in range(self.horizontal_cell_count)
            if (row, col) not in eligible
        ]

    # CNN observation shape: 6 x 5 x 5
    def get_obs(self):
        obs = np.zeros((6, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: agent location
        obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: normalized agent load, broadcast to the whole grid
        obs[1, :, :] = self.agent_load / self.agent_capacity

        # Channel 2: normalized remaining episode steps, broadcast to the whole grid
        obs[2, :, :] = (self.episode_steps - self.step_count) / self.episode_steps

        # Channel 3: item presence
        # Channel 4: normalized item remaining time
        for item_loc, item_time in zip(self.item_locs, self.item_times):
            row, col = item_loc
            time_left = (self.max_response_time - item_time) / self.max_response_time
            obs[3, row, col] = 1.0
            obs[4, row, col] = np.clip(time_left, 0.0, 1.0)

        # Channel 5: blocked cells
        for row, col in self.blocked_cells:
            obs[5, row, col] = 1.0

        return obs
