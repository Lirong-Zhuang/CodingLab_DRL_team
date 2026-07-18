import numpy as np

from environment_v11 import Environment_v11


class Environment_v11h(Environment_v11):
    """Hybrid form of Environment_v11: spatial maps plus two scalars."""

    def __init__(self, variant, data_dir):
        super().__init__(variant, data_dir)
        self.env_name = "11h."

    def get_obs(self):
        full_obs = super().get_obs()
        # Spatial channels: agent, item, time, reachability, heuristic value.
        grid_obs = full_obs[[0, 3, 4, 5, 6], :, :].copy()
        # Non-spatial channels: agent load and remaining episode time.
        scalar_obs = np.array(
            [full_obs[1, 0, 0], full_obs[2, 0, 0]],
            dtype=np.float32,
        )
        return grid_obs, scalar_obs
