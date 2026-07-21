import numpy as np

from environment_v9 import Environment_v9


class Environment_v13(Environment_v9):
    """Variant-2-only Environment v9 with a blocked-cell observation channel."""

    def __init__(self, variant, data_dir, test_data_dir=None):
        if variant != 2:
            raise ValueError("Environment_v13 is only intended for variant 2.")
        super().__init__(variant, data_dir, test_data_dir=test_data_dir)
        self.env_name = "13."
        all_cells = [
            (r, c)
            for r in range(self.vertical_cell_count)
            for c in range(self.horizontal_cell_count)
        ]
        self.blocked_cells = [cell for cell in all_cells if cell not in self.eligible_cells]

    # CNN observation shape: 7 x 5 x 5
    def get_obs(self):
        obs = np.zeros((7, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)

        # Channel 0: agent location
        obs[0, self.agent_loc[0], self.agent_loc[1]] = 1.0

        # Channel 1: agent load, broadcast to whole grid
        obs[1, :, :] = self.agent_load / self.agent_capacity

        # Channel 2: remaining episode steps, broadcast to whole grid
        remaining_steps = (self.episode_steps - self.step_count) / self.episode_steps
        obs[2, :, :] = remaining_steps

        # Channel 3: item presence
        # Channel 4: item remaining time
        # Channel 5: item value based on remaining time minus agent-item distance
        for item_loc, item_time in zip(self.item_locs, self.item_times):
            r, c = item_loc
            time_left = (self.max_response_time - item_time) / self.max_response_time
            time_left = max(0.0, min(1.0, time_left))
            dist_agent_item = self.get_dist_from_coord(self.agent_loc, item_loc)
            remaining_item_steps = self.max_response_time - item_time
            item_value = (remaining_item_steps - dist_agent_item) / self.max_response_time
            item_value = max(0.0, min(1.0, item_value))

            obs[3, r, c] = 1.0
            obs[4, r, c] = time_left
            obs[5, r, c] = item_value

        # Channel 6: blocked cells that the agent is not allowed to enter
        for r, c in self.blocked_cells:
            obs[6, r, c] = 1.0

        return obs
