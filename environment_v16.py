import numpy as np

from environment_v11 import Environment_v11
from spatial_distribution import compute_spawn_distribution


class Environment_v16(Environment_v11):
    """Environment v11 plus an empirical item-spawn probability channel."""

    def __init__(self, variant, data_dir):
        super().__init__(variant, data_dir)
        self.env_name = "16."
        self.spawn_distribution = compute_spawn_distribution(
            variant=self.variant,
            data_dir=self.data_dir,
            normalization="probability",
        )

    # CNN observation shape: 8 x 5 x 5
    def get_obs(self):
        base_obs = super().get_obs()
        obs = np.zeros(
            (8, self.vertical_cell_count, self.horizontal_cell_count),
            dtype=np.float32,
        )
        obs[:7, :, :] = base_obs

        # Channel 7: P(item spawns at this cell), estimated from training data.
        # Values are non-negative and sum to 1 over the grid.
        obs[7, :, :] = self.spawn_distribution
        return obs
