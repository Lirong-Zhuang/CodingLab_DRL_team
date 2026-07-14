from collections import deque

import numpy as np

from environment_v14 import Environment_v14


class Environment_v15(Environment_v14):
    """Environment v14 plus obstacle-aware item reachability."""

    def __init__(self, variant, data_dir):
        super().__init__(variant, data_dir)
        self.env_name = "15."
        self.shortest_path_distances = self._build_shortest_path_distances()

    def _build_shortest_path_distances(self):
        """Precompute graph distances between traversable Variant-2 cells."""
        eligible = set(self.eligible_cells)
        distances = {}

        for start in eligible:
            start_distances = {start: 0}
            queue = deque([start])

            while queue:
                row, col = queue.popleft()
                for neighbor in (
                    (row - 1, col),
                    (row, col + 1),
                    (row + 1, col),
                    (row, col - 1),
                ):
                    if neighbor in eligible and neighbor not in start_distances:
                        start_distances[neighbor] = start_distances[(row, col)] + 1
                        queue.append(neighbor)

            for destination, distance in start_distances.items():
                distances[(start, destination)] = distance

        return distances

    def get_shortest_path_distance(self, start, destination):
        return self.shortest_path_distances.get((start, destination), np.inf)

    # CNN observation shape: 7 x 5 x 5
    def get_obs(self):
        base_obs = super().get_obs()
        obs = np.zeros((7, self.vertical_cell_count, self.horizontal_cell_count), dtype=np.float32)
        obs[:6] = base_obs

        # Channel 6: whether the agent can reach each item before it expires.
        # Distance is the graph shortest path around Variant-2 blocked cells.
        for item_loc, item_time in zip(self.item_locs, self.item_times):
            remaining_item_steps = self.max_response_time - item_time
            distance = self.get_shortest_path_distance(self.agent_loc, item_loc)
            if distance <= remaining_item_steps:
                obs[6, item_loc[0], item_loc[1]] = 1.0

        return obs
