import numpy as np

from environment_v11 import Environment_v11


class Environment_v18(Environment_v11):
    """Variant-2 Environment v11 with an additional blocked-cell channel."""

    def __init__(self, variant, data_dir):
        if variant != 2:
            raise ValueError("Environment_v18 is only intended for variant 2.")

        super().__init__(variant, data_dir)
        self.env_name = "18."

        eligible_cells = set(self.eligible_cells)
        self.blocked_cells = [
            (row, col)
            for row in range(self.vertical_cell_count)
            for col in range(self.horizontal_cell_count)
            if (row, col) not in eligible_cells
        ]

    # CNN observation shape: 8 x 5 x 5
    def get_obs(self):
        base_obs = super().get_obs()
        obs = np.zeros(
            (8, self.vertical_cell_count, self.horizontal_cell_count),
            dtype=np.float32,
        )
        obs[:7, :, :] = base_obs

        # Channel 7: blocked cells that the agent is not allowed to enter.
        for row, col in self.blocked_cells:
            obs[7, row, col] = 1.0

        return obs
